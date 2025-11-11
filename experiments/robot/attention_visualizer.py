"""
Attention Visualization for OpenVLA Patch Relevance

This module provides visualization tools for attention-guided fusion,
including heatmap overlays and patch relevance analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Tuple, Union
import os

class AttentionVisualizer:
    """
    Visualizer for attention-guided fusion patch relevance.
    """
    
    def __init__(self, colormap: str = "hot", alpha: float = 0.6):
        """
        Initialize the attention visualizer.
        
        Args:
            colormap: Matplotlib colormap for heatmap ("hot", "jet", "viridis", etc.)
            alpha: Transparency for heatmap overlay (0.0 = transparent, 1.0 = opaque)
        """
        self.colormap = colormap
        self.alpha = alpha
        
    def patch_relevance_to_heatmap(self, patch_relevance: torch.Tensor) -> np.ndarray:
        """
        Convert 256 patch relevance scores to 16×16 heatmap.
        
        Args:
            patch_relevance: Shape [256] patch relevance scores
            
        Returns:
            np.ndarray: Shape [16, 16] heatmap
        """
        # Convert to numpy
        if torch.is_tensor(patch_relevance):
            relevance_np = patch_relevance.detach().cpu().numpy()
        else:
            relevance_np = patch_relevance
            
        # Reshape to 16×16 grid (224×224 image → 16×16 patches)
        heatmap = relevance_np.reshape(16, 16)
        
        # Normalize to [0, 1] for visualization
        heatmap_min, heatmap_max = heatmap.min(), heatmap.max()
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = np.zeros_like(heatmap)
            
        return heatmap
    
    def resize_heatmap(self, heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize heatmap to match original image size.
        
        Args:
            heatmap: Shape [16, 16] heatmap
            target_size: (height, width) target size
            
        Returns:
            np.ndarray: Resized heatmap
        """
        # Use nearest neighbor interpolation to maintain patch boundaries
        # For ViT: 224×224 image → 16×16 patches (each patch = 14×14 pixels)
        heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_NEAREST)
        return heatmap_resized
    
    def create_heatmap_overlay(
        self, 
        original_image: Union[np.ndarray, Image.Image], 
        patch_relevance: torch.Tensor
    ) -> Image.Image:
        """
        Create heatmap overlay on original image.
        
        Args:
            original_image: Original image (PIL Image or numpy array)
            patch_relevance: Shape [256] patch relevance scores
            
        Returns:
            PIL.Image: Image with heatmap overlay
        """
        # Convert to PIL Image if needed
        if isinstance(original_image, np.ndarray):
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(original_image)
        else:
            pil_image = original_image.copy()
            
        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        # Get image size
        img_width, img_height = pil_image.size
        
        # Convert patch relevance to heatmap
        heatmap = self.patch_relevance_to_heatmap(patch_relevance)
        
        # Resize heatmap to match image size
        heatmap_resized = self.resize_heatmap(heatmap, (img_width, img_height))
        
        # Convert PIL to numpy for proper blending
        img_array = np.array(pil_image)
        
        # Apply colormap
        cmap = cm.get_cmap(self.colormap)
        heatmap_colored = cmap(heatmap_resized)  # Shape: [height, width, 4] (RGBA)
        
        # Extract RGB channels and convert to uint8
        heatmap_rgb = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        
        # Proper alpha blending using cv2.addWeighted
        overlay_array = cv2.addWeighted(
            img_array, 1.0 - self.alpha,  # Original image weight
            heatmap_rgb, self.alpha,       # Heatmap weight
            0                              # Gamma correction
        )
        
        # Convert back to PIL
        overlay = Image.fromarray(overlay_array)
        
        return overlay
    
    def create_side_by_side_visualization(
        self, 
        original_image: Union[np.ndarray, Image.Image], 
        patch_relevance: torch.Tensor,
        attention_mode: str = "action"
    ) -> Image.Image:
        """
        Create side-by-side visualization: original image + heatmap overlay.
        
        Args:
            original_image: Original image
            patch_relevance: Shape [256] patch relevance scores
            attention_mode: Type of attention used ("action", "text", "both")
            
        Returns:
            PIL.Image: Side-by-side visualization
        """
        # Create overlay
        overlay = self.create_heatmap_overlay(original_image, patch_relevance)
        
        # Convert original to PIL if needed
        if isinstance(original_image, np.ndarray):
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)
            original_pil = Image.fromarray(original_image)
        else:
            original_pil = original_image.copy()
            
        if original_pil.mode != 'RGB':
            original_pil = original_pil.convert('RGB')
            
        # Get dimensions
        img_width, img_height = original_pil.size
        
        # Create side-by-side canvas
        canvas_width = img_width * 2 + 20  # 20px gap
        canvas_height = img_height + 60  # 60px for title
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        
        # Paste images
        canvas.paste(original_pil, (0, 50))
        canvas.paste(overlay, (img_width + 20, 50))
        
        # Add titles
        draw = ImageDraw.Draw(canvas)
        try:
            # Try to use a better font
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
            
        # Title text
        title_left = "Original Image"
        title_right = f"Attention Heatmap ({attention_mode})"
        
        # Get text dimensions and center
        left_bbox = draw.textbbox((0, 0), title_left, font=font)
        right_bbox = draw.textbbox((0, 0), title_right, font=font)
        
        left_text_width = left_bbox[2] - left_bbox[0]
        right_text_width = right_bbox[2] - right_bbox[0]
        
        # Draw titles
        draw.text((img_width//2 - left_text_width//2, 20), title_left, fill='black', font=font)
        draw.text((img_width + 20 + img_width//2 - right_text_width//2, 20), title_right, fill='black', font=font)
        
        return canvas
    
    def save_visualization(
        self, 
        image: Image.Image, 
        save_path: str, 
        quality: int = 95
    ):
        """
        Save visualization to file.
        
        Args:
            image: PIL Image to save
            save_path: Path to save the image
            quality: JPEG quality (if saving as JPEG)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save image
        if save_path.lower().endswith('.jpg') or save_path.lower().endswith('.jpeg'):
            image.save(save_path, 'JPEG', quality=quality)
        else:
            image.save(save_path)
            
    def analyze_patch_statistics(self, patch_relevance: torch.Tensor) -> dict:
        """
        Analyze patch relevance statistics.
        
        Args:
            patch_relevance: Shape [256] patch relevance scores
            
        Returns:
            dict: Statistics about patch relevance
        """
        if torch.is_tensor(patch_relevance):
            relevance_np = patch_relevance.detach().cpu().numpy()
        else:
            relevance_np = patch_relevance
            
        stats = {
            'mean': float(relevance_np.mean()),
            'std': float(relevance_np.std()),
            'min': float(relevance_np.min()),
            'max': float(relevance_np.max()),
            'q25': float(np.percentile(relevance_np, 25)),
            'q50': float(np.percentile(relevance_np, 50)),
            'q75': float(np.percentile(relevance_np, 75)),
            'high_attention_patches': int(np.sum(relevance_np > relevance_np.mean() + relevance_np.std())),
            'low_attention_patches': int(np.sum(relevance_np < relevance_np.mean() - relevance_np.std())),
        }
        
        return stats
    
    def create_patch_grid_visualization(
        self, 
        patch_relevance: torch.Tensor, 
        cell_size: int = 30
    ) -> Image.Image:
        """
        Create a 16×16 grid visualization of patch relevance.
        
        Args:
            patch_relevance: Shape [256] patch relevance scores
            cell_size: Size of each cell in pixels
            
        Returns:
            PIL.Image: Grid visualization
        """
        # Convert to heatmap
        heatmap = self.patch_relevance_to_heatmap(patch_relevance)
        
        # Create grid image
        grid_size = 16 * cell_size
        grid_img = Image.new('RGB', (grid_size, grid_size), color='white')
        draw = ImageDraw.Draw(grid_img)
        
        # Apply colormap
        cmap = cm.get_cmap(self.colormap)
        
        # Draw each cell
        for i in range(16):
            for j in range(16):
                # Get relevance value
                relevance = heatmap[i, j]
                
                # Get color from colormap
                color = cmap(relevance)
                rgb_color = tuple(int(c * 255) for c in color[:3])
                
                # Draw cell
                x1, y1 = j * cell_size, i * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size
                draw.rectangle([x1, y1, x2, y2], fill=rgb_color, outline='black', width=1)
                
                # Add relevance value text if cell is large enough
                if cell_size >= 20:
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
                    except:
                        font = ImageFont.load_default()
                    
                    text = f"{relevance:.2f}"
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = x1 + (cell_size - text_width) // 2
                    text_y = y1 + (cell_size - text_height) // 2
                    
                    # Choose text color based on background
                    text_color = 'white' if relevance < 0.5 else 'black'
                    draw.text((text_x, text_y), text, fill=text_color, font=font)
        
        return grid_img
    
    def create_patch_selection_overlay(
        self, 
        original_image: Union[np.ndarray, Image.Image], 
        recompute_mask: torch.Tensor,
        selection_type: str = "recompute",
        color: str = "red",
        alpha: float = 0.3,
        show_grid: bool = True
    ) -> Image.Image:
        """
        Create patch selection overlay showing which patches are recomputed vs reused.
        
        Args:
            original_image: Original image (PIL Image or numpy array)
            recompute_mask: Shape [256] boolean mask (True = recompute, False = reuse)
            selection_type: Type of selection ("recompute", "pixel", "attention", "hybrid")
            color: Color for selected patches ("red", "blue", "green", "purple")
            alpha: Transparency for overlay
            show_grid: Whether to show patch grid lines
            
        Returns:
            PIL.Image: Image with patch selection overlay
        """
        # Convert to PIL Image if needed
        if isinstance(original_image, np.ndarray):
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(original_image)
        else:
            pil_image = original_image.copy()
            
        # Ensure RGB mode
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        # Get image size
        img_width, img_height = pil_image.size
        
        # Convert mask to numpy
        if torch.is_tensor(recompute_mask):
            mask_np = recompute_mask.detach().cpu().numpy()
        else:
            mask_np = recompute_mask
            
        # Reshape to 16x16 grid
        mask_grid = mask_np.reshape(16, 16)
        
        # Create overlay
        overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Color mapping
        color_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255), 
            "green": (0, 255, 0),
            "purple": (128, 0, 128),
            "orange": (255, 165, 0),
            "yellow": (255, 255, 0)
        }
        rgb_color = color_map.get(color.lower(), (255, 0, 0))
        
        # Calculate patch size (224x224 image -> 16x16 patches, each 14x14 pixels)
        patch_size = 14
        
        # Draw selected patches
        for i in range(16):
            for j in range(16):
                if mask_grid[i, j]:  # This patch is selected for recomputation
                    # Calculate patch boundaries
                    x1 = j * patch_size
                    y1 = i * patch_size
                    x2 = x1 + patch_size
                    y2 = y1 + patch_size
                    
                    # Draw filled rectangle with transparency
                    overlay_color = rgb_color + (int(255 * alpha),)
                    draw.rectangle([x1, y1, x2, y2], fill=overlay_color)
                    
                    # Draw border for better visibility
                    border_color = rgb_color + (255,)
                    draw.rectangle([x1, y1, x2, y2], outline=border_color, width=2)
        
        # Optionally draw grid lines
        if show_grid:
            grid_color = (128, 128, 128, 128)  # Semi-transparent gray
            for i in range(1, 16):
                # Vertical lines
                x = i * patch_size
                draw.line([(x, 0), (x, img_height)], fill=grid_color, width=1)
                # Horizontal lines  
                y = i * patch_size
                draw.line([(0, y), (img_width, y)], fill=grid_color, width=1)
        
        # Composite overlay onto original image
        result = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
        return result.convert('RGB')
    
    def create_dual_dimension_comparison(
        self,
        original_image: Union[np.ndarray, Image.Image],
        pixel_diff_values: torch.Tensor,
        attention_relevance: torch.Tensor,
        pixel_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        fusion_mask: torch.Tensor,
        pixel_threshold: float = 0.03,
        attention_top_k: int = 70
    ) -> Image.Image:
        """
        Create comprehensive dual-dimension analysis visualization.
        
        Args:
            original_image: Original RGB image
            pixel_diff_values: Shape [256] pixel difference values  
            attention_relevance: Shape [256] attention relevance scores
            pixel_mask: Shape [256] boolean mask for pixel-based selection
            attention_mask: Shape [256] boolean mask for attention-based selection
            fusion_mask: Shape [256] boolean mask for final hybrid selection
            pixel_threshold: Threshold used for pixel analysis
            attention_top_k: Top-K value used for attention analysis
            
        Returns:
            PIL.Image: Multi-panel comparison visualization
        """
        # Convert original to PIL
        if isinstance(original_image, np.ndarray):
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(original_image)
        else:
            pil_image = original_image.copy()
            
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        img_width, img_height = pil_image.size
        
        # Create individual visualizations
        pixel_heatmap = self.create_heatmap_overlay(pil_image, pixel_diff_values)
        attention_heatmap = self.create_heatmap_overlay(pil_image, attention_relevance)
        pixel_overlay = self.create_patch_selection_overlay(pil_image, pixel_mask, "pixel", "red")
        attention_overlay = self.create_patch_selection_overlay(pil_image, attention_mask, "attention", "blue")
        fusion_overlay = self.create_patch_selection_overlay(pil_image, fusion_mask, "hybrid", "purple")
        
        # Create 3x2 grid layout
        grid_width = img_width * 3 + 40  # 20px gaps between images
        grid_height = img_height * 2 + 100  # 60px for titles + 40px gap
        canvas = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Positions for 3x2 grid
        positions = [
            (0, 60),                                    # Top-left: Original
            (img_width + 20, 60),                      # Top-center: Pixel heatmap  
            (img_width * 2 + 40, 60),                  # Top-right: Attention heatmap
            (0, img_height + 100),                     # Bottom-left: Pixel overlay
            (img_width + 20, img_height + 100),        # Bottom-center: Attention overlay
            (img_width * 2 + 40, img_height + 100)     # Bottom-right: Fusion overlay
        ]
        
        images = [pil_image, pixel_heatmap, attention_heatmap, pixel_overlay, attention_overlay, fusion_overlay]
        # titles = [
        #     "Original Image",
        #     f"Pixel Difference\n(threshold={pixel_threshold})",
        #     f"Attention Relevance\n(top-K={attention_top_k})",
        #     f"Pixel Selection\n({torch.sum(pixel_mask).item()}/256 patches)",
        #     f"Attention Selection\n({torch.sum(attention_mask).item()}/256 patches)", 
        #     f"Hybrid Fusion\n({torch.sum(fusion_mask).item()}/256 recompute)"
        # ]
        titles = [
            "Original Image",
            "Pixel Difference",
            "Attention Relevance",
            "Pixel Selection",
            "Attention Selection",
            "Hybrid Fusion"
        ]
        
        # Paste images and add titles
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
            
        for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
            # Paste image
            canvas.paste(img, pos)
            
            # Add title
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = pos[0] + (img_width - title_width) // 2
            title_y = pos[1] - 40
            
            draw.text((title_x, title_y), title, fill='black', font=font)
        
        return canvas
    
    def create_simplified_comparison(
        self,
        original_image: Union[np.ndarray, Image.Image],
        pixel_diff_values: torch.Tensor,
        attention_relevance: torch.Tensor,
        fusion_mask: torch.Tensor,
        pixel_threshold: float = 0.03,
        attention_top_k: int = 70
    ) -> Image.Image:
        """
        Create simplified 4-panel comparison visualization.
        
        Args:
            original_image: Original RGB image
            pixel_diff_values: Shape [256] pixel difference values  
            attention_relevance: Shape [256] attention relevance scores
            fusion_mask: Shape [256] boolean mask for final hybrid selection
            pixel_threshold: Threshold used for pixel analysis
            attention_top_k: Top-K value used for attention analysis
            
        Returns:
            PIL.Image: 2x2 grid comparison visualization
        """
        # Convert original to PIL
        if isinstance(original_image, np.ndarray):
            if original_image.dtype != np.uint8:
                original_image = (original_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(original_image)
        else:
            pil_image = original_image.copy()
            
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        img_width, img_height = pil_image.size
        
        # Create individual visualizations
        pixel_heatmap = self.create_heatmap_overlay(pil_image, pixel_diff_values)
        attention_heatmap = self.create_heatmap_overlay(pil_image, attention_relevance)
        fusion_overlay = self.create_patch_selection_overlay(pil_image, fusion_mask, "hybrid", "purple", alpha=0.5, show_grid=True)
        
        # Create 2x2 grid layout
        grid_width = img_width * 2 + 20  # 20px gap between images
        grid_height = img_height * 2 + 80  # 60px for titles + 20px gap
        canvas = Image.new('RGB', (grid_width, grid_height), color='white')
        
        # Positions for 2x2 grid
        positions = [
            (0, 50),                                    # Top-left: Original
            (img_width + 20, 50),                      # Top-right: Pixel difference
            (0, img_height + 70),                      # Bottom-left: Attention relevance
            (img_width + 20, img_height + 70)          # Bottom-right: Final selection
        ]
        
        images = [pil_image, pixel_heatmap, attention_heatmap, fusion_overlay]
        titles = [
            "Original Image",
            "Pixel Difference",
            "Attention Relevance", 
            "Final Selection"
        ]
        
        # Paste images and add titles
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        except:
            font = ImageFont.load_default()
            
        for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
            # Paste image
            canvas.paste(img, pos)
            
            # Add title
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_width = title_bbox[2] - title_bbox[0]
            title_x = pos[0] + (img_width - title_width) // 2
            # title_y = pos[1] - 40
            title_y = pos[1] - 17
            
            draw.text((title_x, title_y), title, fill='black', font=font)
        
        return canvas