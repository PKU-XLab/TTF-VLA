"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig
from experiments.robot.aptube_manager import APTubeManager
import time

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    # pixel_values.shape: [1, 6, 224, 224]
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""

        if not self.use_fused_vision_backbone:
            assert False, "We temporarily only support fused vision backbone for now"
        
        # Get or create APTubeManager as a persistent attribute
        if not hasattr(self, 'aptube_manager'):
            self.aptube_manager = APTubeManager()
        manager = self.aptube_manager

        # === Timing starts here, unconditionally ===
        start_time = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        # === Computation block ===
        # These variables will be calculated inside the conditional block
        num_reused_dino = 0
        num_reused_siglip = 0


        """"
        # Key variables for token reuse:
        manager.last_pixel_values.shape: torch.Size([1, 6, 224, 224])
        manager.last_vision_tokens.shape: torch.Size([1, 256, 2176])
        recompute_mask_dino.shape: torch.Size([256])
        recompute_mask_siglip.shape: torch.Size([256])
        pixel_values.shape: torch.Size([1, 6, 224, 224])
        """
        # === AP-TUBE ===
        if not manager.is_enabled():
            img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
            patches_main, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
            patches = torch.cat([patches_main, patches_fused], dim=2)
        else:
            current_shallow_features = None

            # Preparation for semantic/pixel difference
            img_dino, img_siglip = torch.split(pixel_values, [3, 3], dim=1)
            if manager.fusion_mode == "semantic":
                # 一次性获取DINOv2和SigLIP的深、浅两种特征
                # === DINOv2 特征处理 ===
                dino_features_list = self.featurizer.get_intermediate_layers(
                    img_dino, n=[manager.semantic_shallow_layer, len(self.featurizer.blocks) - 2]
                )
                # 直接解包，不做任何切片
                new_tokens_dino, shallow_dino = dino_features_list[1], dino_features_list[0]

                # === SigLIP 特征处理 ===
                siglip_features_list = self.fused_featurizer.get_intermediate_layers(
                    img_siglip, n=[manager.semantic_shallow_layer, len(self.fused_featurizer.blocks) - 2]
                )
                # 直接解包，不做任何切片
                new_tokens_siglip, shallow_siglip = siglip_features_list[1], siglip_features_list[0]

                current_shallow_features = (shallow_dino, shallow_siglip)
                # 计算语义权重，为两种融合模式做准备
                dino_weights, siglip_weights = manager.get_semantic_fusion_weights(shallow_dino, shallow_siglip)
            elif manager.fusion_mode == "pixel":
                new_tokens_dino = self.featurizer(img_dino)
                new_tokens_siglip = self.fused_featurizer(img_siglip)
            elif manager.fusion_mode == "attention":
                # For attention mode, we always compute new features
                # The fusion will be guided by attention weights
                new_tokens_dino = self.featurizer(img_dino)
                new_tokens_siglip = self.fused_featurizer(img_siglip)
            elif manager.fusion_mode == "hybrid":
                # For hybrid mode, we always compute new features
                # The fusion will be guided by both pixel and attention weights
                new_tokens_dino = self.featurizer(img_dino)
                new_tokens_siglip = self.fused_featurizer(img_siglip)
            else:
                assert False, "Invalid fusion mode, please use 'semantic', 'pixel', 'attention', or 'hybrid'!"

            # hard fusion
            if not manager.smooth_fusion_enabled:
                if manager.is_keyframe():
                    img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
                    patches_main, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
                    patches = torch.cat([patches_main, patches_fused], dim=2)
                # TOKEN REUSE CORE LOGIC
                # we first try compute the entire new frame and combine it with the cached tokens, then we try only recomputing the dynamic regions.
                else:
                    # TOKEN REUSE CORE LOGIC
                    # semantic difference
                    if manager.fusion_mode == "semantic":
                        recompute_mask_dino = dino_weights > manager.semantic_threshold
                        recompute_mask_siglip = siglip_weights > manager.semantic_threshold
                    # pixel difference
                    elif manager.fusion_mode == "pixel":
                        recompute_mask_dino, recompute_mask_siglip = manager.get_pixel_recompute_mask(pixel_values)
                    # attention-guided difference
                    elif manager.fusion_mode == "attention":
                        recompute_mask_dino, recompute_mask_siglip = manager.get_attention_recompute_mask()
                    # hybrid difference (pixel + attention)
                    elif manager.fusion_mode == "hybrid":
                        recompute_mask_dino, recompute_mask_siglip = manager.get_hybrid_recompute_mask(pixel_values)
                    else:
                        assert False, "Invalid fusion mode, please use 'semantic', 'pixel', 'attention', or 'hybrid'!"
                    
                    num_reused_dino = (~recompute_mask_dino).sum().item()
                    num_reused_siglip = (~recompute_mask_siglip).sum().item()
                    
                    # Log attention-guided fusion info
                    if manager.fusion_mode == "attention" and manager.step_counter % 10 == 0:
                        print(f"  Vision backbone - VLA-Cache attention fusion")
                        print(f"  DINO reused: {num_reused_dino}/256, SigLIP reused: {num_reused_siglip}/256")

                    # 1. Separate the cached tokens for DINO and SigLIP
                    # DINO tokens are the first 1024 dimensions, SigLIP are the next 1152.
                    cached_tokens_dino, cached_tokens_siglip = torch.split(
                        manager.last_vision_tokens, [1024, 1152], dim=2
                    )

                    # 3. Initialize final tokens with the cached versions
                    final_tokens_dino = cached_tokens_dino.clone()
                    final_tokens_siglip = cached_tokens_siglip.clone()
                    
                    # 4. Overwrite the dynamic regions with the newly computed tokens
                    final_tokens_dino[0, recompute_mask_dino] = new_tokens_dino[0, recompute_mask_dino]
                    final_tokens_siglip[0, recompute_mask_siglip] = new_tokens_siglip[0, recompute_mask_siglip]

                    # 5. Concatenate the final DINO and SigLIP tokens
                    patches = torch.cat([final_tokens_dino, final_tokens_siglip], dim=2)
            # smooth fusion
            else:                
                if manager.is_keyframe():
                    img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
                    patches_main, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
                    patches = torch.cat([patches_main, patches_fused], dim=2)
                # TOKEN REUSE CORE LOGIC
                # we first try compute the entire new frame and combine it with the cached tokens, then we try only recomputing the dynamic regions.
                else:
                    if manager.fusion_mode == "semantic":
                        dino_fusion_weights, siglip_fusion_weights = dino_weights, siglip_weights
                    elif manager.fusion_mode == "pixel":
                        dino_fusion_weights, siglip_fusion_weights = manager.get_fusion_weights(pixel_values)
                    elif manager.fusion_mode == "attention":
                        dino_fusion_weights, siglip_fusion_weights = manager.get_attention_fusion_weights()
                    else:
                        assert False, "Invalid fusion mode, please use 'semantic', 'pixel', or 'attention'!"
                    
                    num_reused_dino = (1 - dino_fusion_weights).sum().item()
                    num_reused_siglip = (1 - siglip_fusion_weights).sum().item()

                    # 1. Separate the cached tokens for DINO and SigLIP
                    # DINO tokens are the first 1024 dimensions, SigLIP are the next 1152.
                    cached_tokens_dino, cached_tokens_siglip = torch.split(
                        manager.last_vision_tokens, [1024, 1152], dim=2
                    )

                    # 3. Prepare weights for broadcasting
                    # Shape: (num_patches,) -> (1, num_patches, 1)
                    dino_weights = dino_fusion_weights.unsqueeze(0).unsqueeze(-1)
                    siglip_weights = siglip_fusion_weights.unsqueeze(0).unsqueeze(-1)
                    # Convert weights to the same dtype as tokens to prevent mismatch
                    dino_weights = dino_weights.to(new_tokens_dino.dtype)
                    siglip_weights = siglip_weights.to(new_tokens_siglip.dtype)

                    # 4. Perform smooth fusion using weighted average
                    final_tokens_dino = (dino_weights * new_tokens_dino) + ((1 - dino_weights) * cached_tokens_dino)
                    final_tokens_siglip = (siglip_weights * new_tokens_siglip) + ((1 - siglip_weights) * cached_tokens_siglip)

                    # 5. Concatenate the final DINO and SigLIP tokens
                    patches = torch.cat([final_tokens_dino, final_tokens_siglip], dim=2)
                


        # === Timing ends here, unconditionally ===
        end_event.record()
        torch.cuda.synchronize()

        # === Unconditional Metrics Storage ===
        # Store timing
        manager.metrics["vision_wall_time_ms"].append((time.perf_counter() - start_time) * 1000)
        manager.metrics["vision_cuda_time_ms"].append(start_event.elapsed_time(end_event))

        # Store FLOPs and reuse stats
        total_patches_per_backbone = self.featurizer.patch_embed.num_patches    # This is 256
        if manager.is_enabled():
            gflops_saved_dino = (num_reused_dino / total_patches_per_backbone) * manager.baseline_dino_gflops
            gflops_saved_siglip = (num_reused_siglip / total_patches_per_backbone) * manager.baseline_siglip_gflops
            current_vision_gflops = (manager.baseline_dino_gflops + manager.baseline_siglip_gflops) - (gflops_saved_dino + gflops_saved_siglip)
            manager.metrics["vision_gflops"].append(current_vision_gflops)
            manager.metrics["reused_patches_dino"].append(num_reused_dino)
            manager.metrics["reused_patches_siglip"].append(num_reused_siglip)
            manager.metrics["reused_patches_total"].append(num_reused_dino + num_reused_siglip)
            manager.metrics["total_patches"].append(total_patches_per_backbone * 2)
        else:
            # For baseline, store the baseline GFLOPs and 0 for reuse
            total_vision_baseline = manager.baseline_dino_gflops + manager.baseline_siglip_gflops
            manager.metrics["vision_gflops"].append(total_vision_baseline)
            manager.metrics["reused_patches_dino"].append(0)
            manager.metrics["reused_patches_siglip"].append(0)
            manager.metrics["reused_patches_total"].append(0)
            manager.metrics["total_patches"].append(total_patches_per_backbone * 2)

        # Update cache if enabled, after all computations for this step are done
        if manager.is_enabled():
            manager.update_cache(pixel_values=pixel_values, vision_tokens=patches, shallow_features=current_shallow_features)

        return patches


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLAForActionPrediction(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

    """
    input_ids.shape = torch.Size([1, 25])
    input_ids[0] = tensor([1,   512, 29901,  1724,  3158,   881,   278, 19964,  2125,   304,
         5839,   701,   278, 22968, 22300,   322,  2058,   372,   297,   278,
        25972, 29973,    13,  3744, 29901], device='cuda:0')
    unnorm_key = 'libero_xxx', xxx is the name of the task
    kwargs.keys() = dict_keys(['attention_mask', 'pixel_values', 'do_sample'])
    kwargs['attention_mask'].shape = torch.Size([1, 25])
    kwargs['pixel_values'].shape = torch.Size([1, 6, 224, 224])
    kwargs['do_sample'] = False
    """
    def predict_action(
        self, input_ids: Optional[torch.LongTensor] = None, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """Thin wrapper around .generate() that decodes predicted actions and unnormalizes them."""
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        """
        generated_ids.shape =torch.Size([1, 33])
        """
        # Get or create APTubeManager - use the same instance as vision backbone
        if hasattr(self.vision_backbone, 'aptube_manager'):
            manager = self.vision_backbone.aptube_manager
        else:
            from experiments.robot.aptube_manager import APTubeManager
            manager = APTubeManager()
            self.vision_backbone.aptube_manager = manager
            
        need_attention = manager.aptube_enabled and manager.fusion_mode in ["attention", "hybrid"]
        
        # Save input_ids_len for dynamic token position calculation
        manager.input_ids_len = input_ids.shape[1]
        
        if need_attention:
            # Run VLA inference with attention extraction
            generation_output = self.generate(
                input_ids, 
                max_new_tokens=self.get_action_dim(unnorm_key), 
                output_attentions=True,
                return_dict_in_generate=True,
                **kwargs
            )
            
            # Extract generated sequences and attention weights
            generated_ids = generation_output.sequences
            attentions = generation_output.attentions
            
            # Store attention weights for attention-guided fusion
            # Analyze all 7 action tokens' attention patterns
            print(f"  DEBUG: attentions available: {attentions is not None}")
            if attentions and len(attentions) > 0:
                print(f"  DEBUG: attentions length: {len(attentions)}")
                print(f"  DEBUG: first step type: {type(attentions[0])}")
                
                # attentions is a tuple of tuples: (step1_layers, step2_layers, ...)
                # Each step contains attention from all layers
                
                # Extract attention weights from all 7 action tokens
                all_step_attentions = []
                
                for step_idx, step_attention in enumerate(attentions):
                    if isinstance(step_attention, (tuple, list)):
                        print(f"  DEBUG: step {step_idx} has {len(step_attention)} layers")
                        
                        # Try different layers to find the right format
                        for layer_idx in [15, -1, 0]:  # Try layer 15 (VLA-Cache), last layer, first layer
                            layer_idx = layer_idx if layer_idx >= 0 else len(step_attention) + layer_idx
                            if 0 <= layer_idx < len(step_attention):
                                candidate_attention = step_attention[layer_idx]
                                print(f"  DEBUG: step {step_idx} layer {layer_idx} shape: {candidate_attention.shape}")
                                
                                # Look for 4D attention tensor [batch, heads, seq, seq]
                                if candidate_attention.dim() == 4 and candidate_attention.shape[2] >= 257:
                                    all_step_attentions.append(candidate_attention)
                                    print(f"  DEBUG: stored step {step_idx} layer {layer_idx} attention")
                                    break
                        else:
                            print(f"  DEBUG: no valid attention found in step {step_idx}")
                    else:
                        print(f"  DEBUG: step {step_idx} is not tuple/list: {type(step_attention)}")
                
                # Store VLA-Cache style attention: attentions[0] (first step's all layers)
                manager.last_attention_layers = attentions[0]  
                print(f"  DEBUG: stored VLA-Cache attention layers: {len(attentions[0])} layers")
        else:
            # Run VLA inference without attention extraction (faster)
            # Let output_attentions=True to make sure fair comparison
            generation_output = self.generate(
                input_ids, 
                max_new_tokens=self.get_action_dim(unnorm_key), 
                output_attentions=True,
                return_dict_in_generate=True,
                **kwargs
            )
            generated_ids = generation_output.sequences

        """
        generation_output = self.generate(
            input_ids, 
            max_new_tokens=self.get_action_dim(unnorm_key), 
            output_attentions=True,             # <--- 激活注意力
            return_dict_in_generate=True,       # <--- 改变返回值类型
            **kwargs
        )
        
        # Manually extract the data from the new output object
        generated_ids = generation_output.sequences
        attentions = generation_output.attentions

        # 调试最需要的信息
        # Number of generation steps: 7
        # Number of layers in the model: 32
        # Shape of one attention tensor: torch.Size([1, 32, 282, 282])
        print(f"Number of generation steps: {len(attentions)}")
        print(f"Number of layers in the model: {len(attentions[0])}")
        print(f"Shape of one attention tensor: {attentions[0][0].shape}")
        """

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[0, -self.get_action_dim(unnorm_key) :].cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
