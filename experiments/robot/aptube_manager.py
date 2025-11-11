from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

class APTubeManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(APTubeManager, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Configuration
        self.aptube_enabled = False
        self.baseline_dino_gflops = 0.0
        self.baseline_siglip_gflops = 0.0
        self.patch_diff_threshold = 0.1
        self.keyframe_interval = 5
        self.smooth_fusion_enabled = False
        self.fusion_mode = "pixel"  # Options: "pixel", "semantic", "attention", "hybrid"
        self.semantic_shallow_layer = 2
        self.semantic_threshold = 0.5
        self.visualize_attention = False  # Generate attention heatmap visualizations
        self.visualization_save_dir = "./attention_visualizations"  # Directory to save visualizations
        self.visualization_interval = 30  # Generate visualization every N steps
        self.PRINT_INTERVAL = 3
        # State
        self.step_counter = 0
        self.last_pixel_values: Optional[torch.Tensor] = None
        self.last_vision_tokens: Optional[torch.Tensor] = None
        self.last_shallow_dino_features: Optional[torch.Tensor] = None
        self.last_shallow_siglip_features: Optional[torch.Tensor] = None
        # VLA-Cache style attention storage
        self.last_attention_layers: Optional[tuple] = None  # Store attentions[0] directly
        self.input_ids_len: Optional[int] = None
        self.last_pixel_diff_values: Optional[torch.Tensor] = None  # For pixel difference visualization

        # Metrics
        self.metrics = {
            "vision_wall_time_ms": [],
            "vision_cuda_time_ms": [],
            "vision_gflops": [],          # Will store the final, accurately calculated total vision GFLOPs
            "reused_patches_dino": [],    # New: For DINO's reuse count
            "reused_patches_siglip": [],  # New: For SigLIP's reuse count
            "reused_patches_total": [],   # This will store total reused SLOTS (dino + siglip)
            "total_patches": [],          # This will store total available SLOTS (512)
        }
        
        # Attention-guided fusion system
        self.attention_mode = "text"  # Options: "text", "action"
        self.use_multi_layer = False  # Whether to use multi-layer attention aggregation

    def configure(self, aptube_enabled: bool = False, **kwargs):
        self.aptube_enabled = aptube_enabled
        self.baseline_dino_gflops = kwargs.get('baseline_dino_gflops', 0.0)
        self.baseline_siglip_gflops = kwargs.get('baseline_siglip_gflops', 0.0)
        self.patch_diff_threshold = kwargs.get('patch_diff_threshold', 0.1)
        self.keyframe_interval = kwargs.get('keyframe_interval', 5)
        self.smooth_fusion_enabled = kwargs.get('smooth_fusion_enabled', False)
        self.fusion_mode = kwargs.get('fusion_mode', "pixel")
        self.semantic_shallow_layer = kwargs.get('semantic_shallow_layer', 2)
        self.semantic_threshold = kwargs.get('semantic_threshold', 0.5)
        # VLA-Cache style attention parameters
        self.attention_layer_id = kwargs.get('attention_layer_id', 15)
        self.attention_top_k = kwargs.get('attention_top_k', 120)
        self.attention_mode = kwargs.get('attention_mode', 'text')
        self.use_multi_layer = kwargs.get('use_multi_layer', False)
        self.visualize_attention = kwargs.get('visualize_attention', False)
        self.visualization_save_dir = kwargs.get('visualization_save_dir', "./attention_visualizations")
        self.visualization_interval = kwargs.get('visualization_interval', 30)
        self.reset_metrics()
        self.reset_state()

    def reset_state(self):
        self.step_counter = 0
        self.last_pixel_values = None
        self.last_vision_tokens = None
        self.last_pixel_diff_values = None

    def reset_metrics(self):
        for key in self.metrics:
            self.metrics[key].clear()

    def is_enabled(self) -> bool:
        return self.aptube_enabled
    
    def is_keyframe(self) -> bool:
        return self.step_counter == 0 or self.last_pixel_values is None or self.step_counter % self.keyframe_interval == 0

    def get_pixel_recompute_mask(self, current_pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:       
        # Must be a keyframe if it's the first step or no cache exists or it's a keyframe interval
        if self.is_keyframe():
            # For the first frame, all patches must be computed.
            recompute_mask_dino = torch.ones(256, dtype=torch.bool, device=current_pixel_values.device)
            recompute_mask_siglip = torch.ones(256, dtype=torch.bool, device=current_pixel_values.device)
            return recompute_mask_dino, recompute_mask_siglip

        
        def _calculate_mask_for_one_part_vision_encoder(prev_img_part, curr_img_part):
            # --- Time-difference patch selection ---
            img_prev = prev_img_part.to(torch.float32)
            img_curr = curr_img_part.to(torch.float32)

            # Convert to grayscale
            weights = torch.tensor([0.299, 0.587, 0.114], device=img_curr.device).view(1, 3, 1, 1)
            gray_prev = torch.sum(img_prev * weights, dim=1, keepdim=True)
            gray_curr = torch.sum(img_curr * weights, dim=1, keepdim=True)

            # Calculate pixel-wise absolute difference
            diff = torch.abs(gray_prev - gray_curr)

            # Calculate average difference per patch (14x14)
            # Input shape: [1, 1, 224, 224] -> Output shape: [1, 1, 16, 16]
            patch_avg_diff = F.avg_pool2d(diff, kernel_size=14, stride=14)

            # Return both the mask and the raw diff values for visualization
            diff_values = patch_avg_diff.flatten()
            mask = diff_values > self.patch_diff_threshold
            return mask, diff_values


        # Calculate mask and diff values for DINO (first 3 channels)
        recompute_mask_dino, diff_values_dino = _calculate_mask_for_one_part_vision_encoder(
            self.last_pixel_values[:, :3, :, :], current_pixel_values[:, :3, :, :]
        )

        # Calculate mask and diff values for SigLIP (last 3 channels)
        recompute_mask_siglip, diff_values_siglip = _calculate_mask_for_one_part_vision_encoder(
            self.last_pixel_values[:, 3:, :, :], current_pixel_values[:, 3:, :, :]
        )

        # Save diff values for visualization (use DINO as representative)
        self.last_pixel_diff_values = diff_values_dino

        return recompute_mask_dino, recompute_mask_siglip


    def get_fusion_weights(self, current_pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:    
        # Must be a keyframe if it's the first step or no cache exists or it's a keyframe interval
        if self.is_keyframe():
            # For a keyframe, the weight for new features is 1.0 (i.e., use 100% new features)
            dino_fusion_weights = torch.ones(256, dtype=torch.float32, device=current_pixel_values.device)
            siglip_fusion_weights = torch.ones(256, dtype=torch.float32, device=current_pixel_values.device)
            return dino_fusion_weights, siglip_fusion_weights

        def _calculate_weights_for_one_part_vision_encoder(prev_img_part, curr_img_part):
            # --- Time-difference patch selection ---
            img_prev = prev_img_part.to(torch.float32)
            img_curr = curr_img_part.to(torch.float32)

            # Convert to grayscale
            weights = torch.tensor([0.299, 0.587, 0.114], device=img_curr.device).view(1, 3, 1, 1)
            gray_prev = torch.sum(img_prev * weights, dim=1, keepdim=True)
            gray_curr = torch.sum(img_curr * weights, dim=1, keepdim=True)

            # Calculate pixel-wise absolute difference
            diff = torch.abs(gray_prev - gray_curr)

            # Calculate average difference per patch (14x14)
            # Input shape: [1, 1, 224, 224] -> Output shape: [1, 1, 16, 16]
            patch_avg_diff = F.avg_pool2d(diff, kernel_size=14, stride=14)

            # Convert patch differences to a [0, 1] weight.
            # This weight represents how much to trust the new features.
            fusion_weights = (patch_avg_diff.flatten() / self.patch_diff_threshold).clamp(0.0, 1.0)
            
            return fusion_weights
        
        # Calculate weights for DINO (first 3 channels)
        dino_fusion_weights = _calculate_weights_for_one_part_vision_encoder(
            self.last_pixel_values[:, :3, :, :], current_pixel_values[:, :3, :, :]
        )

        # Calculate weights for SigLIP (last 3 channels)
        siglip_fusion_weights = _calculate_weights_for_one_part_vision_encoder(
            self.last_pixel_values[:, 3:, :, :], current_pixel_values[:, 3:, :, :]
        )

        return dino_fusion_weights, siglip_fusion_weights


    def get_semantic_fusion_weights(self, new_shallow_dino, new_shallow_siglip):
        """Calculates fusion weights based on cosine similarity of shallow features."""
        # 如果是第一帧，返回全1权重，表示全部更新
        if self.last_shallow_dino_features is None or self.last_shallow_siglip_features is None:
            w_dino = torch.ones(new_shallow_dino.shape[1], device=new_shallow_dino.device)
            w_siglip = torch.ones(new_shallow_siglip.shape[1], device=new_shallow_siglip.device)
            return w_dino, w_siglip

        # DINO
        sim_dino = F.cosine_similarity(new_shallow_dino, self.last_shallow_dino_features, dim=-1).squeeze(0)
        w_dino = 1.0 - sim_dino.clamp(0, 1)

        # SigLIP
        sim_siglip = F.cosine_similarity(new_shallow_siglip, self.last_shallow_siglip_features, dim=-1).squeeze(0)
        w_siglip = 1.0 - sim_siglip.clamp(0, 1)

        return w_dino, w_siglip


    def update_cache(self, pixel_values: torch.Tensor, vision_tokens: torch.Tensor, shallow_features: Tuple[torch.Tensor, torch.Tensor]):
        if self.fusion_mode == "pixel":
            self.last_pixel_values = pixel_values.detach().clone()
            self.last_vision_tokens = vision_tokens.detach().clone()
            self.step_counter += 1
        elif self.fusion_mode == "semantic" and shallow_features is not None:
            self.last_pixel_values = pixel_values.detach().clone()
            dino_feats, siglip_feats = shallow_features
            self.last_shallow_dino_features = dino_feats.detach().clone()
            self.last_shallow_siglip_features = siglip_feats.detach().clone()
            self.step_counter += 1
        elif self.fusion_mode == "attention":
            self.last_pixel_values = pixel_values.detach().clone()
            self.last_vision_tokens = vision_tokens.detach().clone()
            self.step_counter += 1
        elif self.fusion_mode == "hybrid":
            self.last_pixel_values = pixel_values.detach().clone()
            self.last_vision_tokens = vision_tokens.detach().clone()
            self.step_counter += 1
        else:
            raise ValueError(f"Invalid fusion mode: {self.fusion_mode}")
            
    # VLA-Cache style attention methods
    def token_attention_merge(self, multihead_attention, layer_ids=None, attention_mode="text", use_multi_layer=False):
        """
        Extract token-to-vision attention using configurable mode and layer aggregation.
        
        Args:
            multihead_attention: tuple of layers from attentions[0] (33 layers total)
            layer_ids: list of attention layers to aggregate (default from config)
            attention_mode: "text" or "action" to specify which tokens to use
            use_multi_layer: whether to aggregate across multiple layers
            
        Returns:
            torch.Tensor: shape [256], patch relevance scores
        """
        if layer_ids is None:
            if use_multi_layer:
                # Use multiple layers for aggregation around middle layers (total 33 layers)
                layer_ids = [10, 15, 20, 25]
            else:
                layer_ids = [self.attention_layer_id]  # Single layer (default 15)
            
        # OpenVLA token positions
        v_token_start = 1        # Vision tokens: positions 1-256
        v_token_end = 257        # 
        t_token_start = 257      # Text tokens start after vision
        t_token_end = 292        # Max text tokens (support up to 35)
        a_token_start = t_token_end  # Action tokens start after text
        a_token_end = a_token_start + 7  # 7 action dimensions
        
        aggregated_scores = None
        
        for layer_id in layer_ids:
            if layer_id >= len(multihead_attention):
                continue
                
            # Select layer and average across heads: [1, 32, seq_len, seq_len] -> [seq_len, seq_len]
            attn_map = multihead_attention[layer_id].to(torch.float32).squeeze(0).mean(dim=0)
            
            layer_scores = torch.zeros(256, device=attn_map.device)
            
            if attention_mode == "text":
                # Text-to-vision attention
                if t_token_start < attn_map.shape[0]:
                    seq_len = attn_map.shape[0]
                    actual_t_end = min(t_token_end, seq_len)
                    if actual_t_end > t_token_start:
                        text_to_vision = attn_map[t_token_start:actual_t_end, v_token_start:v_token_end]
                        layer_scores = text_to_vision.mean(dim=0)  # Average across text tokens
                        
            elif attention_mode == "action":
                # Action-to-vision attention (first action token only for task relevance)
                if a_token_start < attn_map.shape[0]:
                    action_to_vision = attn_map[a_token_start, v_token_start:v_token_end]
                    layer_scores = action_to_vision
            
            # Aggregate across layers
            if aggregated_scores is None:
                aggregated_scores = layer_scores
            else:
                aggregated_scores += layer_scores
        
        # Average across layers if using multiple layers
        if len(layer_ids) > 1:
            aggregated_scores = aggregated_scores / len(layer_ids)
            
        result = aggregated_scores.cpu()
        print(f"    DEBUG: Attention computed, mode: {attention_mode}, layers: {layer_ids}, use_multi_layer: {use_multi_layer}, shape: {result.shape}")
        return result

    def get_top_attention_patches(self, attn_scores, top_k=None):
        """
        Select top-k patch indices based on attention scores using VLA-Cache method.
        
        Args:
            attn_scores: [256] patch relevance scores
            top_k: number of patches to select (default from config)
            
        Returns:
            list: top-k patch indices
        """
        if top_k is None:
            top_k = self.attention_top_k
            
        # Convert to numpy if needed
        attn_scores = attn_scores.cpu().numpy() if isinstance(attn_scores, torch.Tensor) else attn_scores
        
        # Reshape to 16x16 grid (256 patches)
        attn = attn_scores.reshape(16, 16)
        
        # Optional resize (keep same size for now)
        import cv2
        attn_resized = cv2.resize(attn, (16, 16))
        
        # Create (index, score) pairs
        flat = [(i * 16 + j, attn_resized[i, j]) for i in range(16) for j in range(16)]
        
        # Sort by score descending
        flat.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k indices
        return [idx for idx, _ in flat[:top_k]]

    def get_attention_recompute_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get recompute mask based on VLA-Cache style attention analysis."""
        if self.last_attention_layers is None:
            # No attention data available, recompute all
            device = self.last_pixel_values.device if self.last_pixel_values is not None else 'cpu'
            return torch.ones(256, dtype=torch.bool, device=device), torch.ones(256, dtype=torch.bool, device=device)
        
        try:
            # Extract patch relevance using VLA-Cache method
            print(f"  DEBUG: Starting VLA-Cache attention processing...")
            patch_relevance = self.token_attention_merge(
                self.last_attention_layers, 
                attention_mode=self.attention_mode,
                use_multi_layer=self.use_multi_layer
            )
            print(f"  DEBUG: Patch relevance shape: {patch_relevance.shape}, min: {patch_relevance.min():.4f}, max: {patch_relevance.max():.4f}")
            
            # Get top-k important patches
            important_patch_indices = self.get_top_attention_patches(patch_relevance)
            print(f"  DEBUG: Selected {len(important_patch_indices)} important patches (top-{self.attention_top_k})")
            print(f"  DEBUG: Important patch indices (first 10): {important_patch_indices[:10]}")
            
            # Create recompute mask: True for important patches (must recompute), False for others (can reuse)
            device = self.last_pixel_values.device if self.last_pixel_values is not None else 'cpu'
            recompute_mask = torch.zeros(256, dtype=torch.bool, device=device)  # Start with all reuse
            
            # Important patches need recomputation (use new tokens)
            for idx in important_patch_indices:
                recompute_mask[idx] = True
            
            reuse_count = (recompute_mask == False).sum().item()
            recompute_count = (recompute_mask == True).sum().item() 
            print(f"  DEBUG: Will recompute {recompute_count}/256 task-relevant patches, reuse {reuse_count}/256 patches ({reuse_count/256*100:.1f}% reuse)")
            
            # Both encoders use same mask for now
            return recompute_mask, recompute_mask.clone()
            
        except Exception as e:
            # Fallback to recompute all if attention processing fails
            print(f"  DEBUG: Attention processing failed: {e}")
            import traceback
            traceback.print_exc()
            device = self.last_pixel_values.device if self.last_pixel_values is not None else 'cpu'
            return torch.ones(256, dtype=torch.bool, device=device), torch.ones(256, dtype=torch.bool, device=device)
    
    
    
    def get_hybrid_recompute_mask(self, current_pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get hybrid recompute mask combining pixel and attention methods.
        
        Logic: Reuse patches that are BOTH static AND task-irrelevant
               Recompute patches that are EITHER dynamic OR task-relevant
        
        Args:
            current_pixel_values: Current frame pixel values
            
        Returns:
            Tuple of recompute masks for DINO and SigLIP
        """
        try:
            # Get pixel-based recompute mask (True = dynamic, False = static)
            pixel_recompute_dino, pixel_recompute_siglip = self.get_pixel_recompute_mask(current_pixel_values)
            pixel_dynamic_count = pixel_recompute_dino.sum().item()
            
            # Get attention-based recompute mask (True = task-relevant, False = task-irrelevant)  
            attention_recompute_dino, attention_recompute_siglip = self.get_attention_recompute_mask()
            attention_relevant_count = attention_recompute_dino.sum().item()
            
            # Hybrid logic: recompute = pixel_recompute OR attention_recompute
            # This means we reuse only patches that are BOTH static AND task-irrelevant
            hybrid_recompute_dino = pixel_recompute_dino | attention_recompute_dino
            hybrid_recompute_siglip = pixel_recompute_siglip | attention_recompute_siglip
            
            reuse_dino = (~hybrid_recompute_dino).sum().item()
            reuse_siglip = (~hybrid_recompute_siglip).sum().item()
            hybrid_reuse_rate = (reuse_dino + reuse_siglip) / 512 * 100
            
            # if self.step_counter % self.PRINT_INTERVAL == 0:
            if True:
                print(f"  DEBUG: Hybrid Analysis:")
                print(f"    Pixel: {pixel_dynamic_count}/256 dynamic patches")
                print(f"    Attention: {attention_relevant_count}/256 task-relevant patches")  
                print(f"    Hybrid: DINO {reuse_dino}/256 reuse, SigLIP {reuse_siglip}/256 reuse")
                print(f"    Combined reuse rate: {hybrid_reuse_rate:.1f}%")
            
            return hybrid_recompute_dino, hybrid_recompute_siglip
            
        except Exception as e:
            print(f"  DEBUG: Hybrid fusion failed: {e}, fallback to pixel-based fusion")
            # Fallback to pixel-based fusion
            return self.get_pixel_recompute_mask(current_pixel_values)
    
    
    def get_pixel_diff_values_for_visualization(self) -> Optional[torch.Tensor]:
        """
        Get pixel difference values for visualization purposes.
        
        Returns:
            torch.Tensor: Shape [256] pixel difference values, or None if not available
        """
        return self.last_pixel_diff_values
    
    def get_patch_relevance_for_visualization(self) -> Optional[torch.Tensor]:
        """
        Get patch relevance scores for attention visualization.
        
        Returns:
            torch.Tensor: Shape [256] patch relevance scores, or None if not available
        """
        if not self.aptube_enabled or self.fusion_mode not in ["attention", "hybrid"]:
            return None
            
        if self.last_attention_layers is None:
            return None
            
        try:
            # Use the same method as get_attention_recompute_mask to extract patch relevance
            patch_relevance = self.token_attention_merge(
                self.last_attention_layers, 
                attention_mode=self.attention_mode,
                use_multi_layer=self.use_multi_layer
            )
            
            return patch_relevance  # Shape [256]
            
        except Exception as e:
            print(f"Warning: Failed to get patch relevance for visualization: {e}")
            return None
    
    def get_patch_selection_masks_for_visualization(self) -> Optional[dict]:
        """
        Get patch selection masks for comprehensive visualization.
        
        Returns:
            dict: Contains pixel_mask, attention_mask, and fusion_mask, or None if not available
        """
        if not self.aptube_enabled or self.last_pixel_values is None:
            return None
            
        try:
            # Get current pixel values (dummy for demonstration - in real use this comes from model forward)
            current_pixel_values = self.last_pixel_values  # This should be the current frame
            
            # Get individual masks
            if self.fusion_mode == "pixel":
                pixel_mask_dino, pixel_mask_siglip = self.get_pixel_recompute_mask(current_pixel_values)
                return {
                    'pixel_mask': pixel_mask_dino,  # Use DINO mask for visualization
                    'attention_mask': None,
                    'fusion_mask': pixel_mask_dino
                }
            elif self.fusion_mode == "attention":
                attention_mask_dino, attention_mask_siglip = self.get_attention_recompute_mask()
                return {
                    'pixel_mask': None,
                    'attention_mask': attention_mask_dino,  # Use DINO mask for visualization
                    'fusion_mask': attention_mask_dino
                }
            elif self.fusion_mode == "hybrid":
                # Get both masks
                pixel_mask_dino, pixel_mask_siglip = self.get_pixel_recompute_mask(current_pixel_values) 
                attention_mask_dino, attention_mask_siglip = self.get_attention_recompute_mask()
                
                if pixel_mask_dino is not None and attention_mask_dino is not None:
                    # Apply OR logic for hybrid fusion
                    fusion_mask = pixel_mask_dino | attention_mask_dino
                    
                    return {
                        'pixel_mask': pixel_mask_dino,
                        'attention_mask': attention_mask_dino, 
                        'fusion_mask': fusion_mask
                    }
            
            return None
            
        except Exception as e:
            print(f"Warning: Failed to get patch selection masks: {e}")
            return None