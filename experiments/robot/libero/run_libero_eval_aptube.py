"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark
import time
import torch

import wandb

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from experiments.robot.aptube_manager import APTubeManager


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    num_tasks: Optional[int] = None                  # Number of tasks to evaluate from the suite. If None, all tasks are evaluated.
    task_start_id: int = 0                           # Starting task ID (0-based indexing)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on

    #################################################################################################################
    # AP-TUBE parameters
    #################################################################################################################
    aptube_enabled: bool = False  # Enable AP-TUBE
    baseline_dino_gflops: float = 158.0496  # Baseline GFLOPs for DINOv2
    baseline_siglip_gflops: float = 210.9423  # Baseline GFLOPs for SigLIP
    baseline_vision_total_gflops: float = 368.9919  # Baseline GFLOPs for total vision
    baseline_e2e_gflops: float = 3894.2080  # Baseline GFLOPs for end-to-end
    patch_diff_threshold: float = 0.1  # Threshold for patch difference
    keyframe_interval: int = 5  # Keyframe interval
    smooth_fusion_enabled: bool = False  # Enable smooth fusion
    fusion_mode: str = "pixel"  # Fusion mode
    semantic_shallow_layer: int = 2  # Semantic shallow layer
    semantic_threshold: float = 0.5  # Threshold for semantic fusion
    attention_layer_id: int = 15  # VLA-Cache style attention layer ID
    attention_top_k: int = 120  # VLA-Cache style top-k patches to select
    visualize_attention: bool = False  # Generate attention heatmap visualizations
    visualization_save_dir: str = "./attention_visualizations"  # Directory to save visualizations
    visualization_interval: int = 30  # Generate visualization every N steps
    attention_mode: str = "text"  # Attention mode: "text" or "action"
    use_multi_layer: bool = False  # Use multi-layer attention aggregation


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # Initialize AP-TUBE Manager
    manager = APTubeManager()
    manager.configure(
        aptube_enabled=cfg.aptube_enabled,
        baseline_dino_gflops=cfg.baseline_dino_gflops,
        baseline_siglip_gflops=cfg.baseline_siglip_gflops,
        patch_diff_threshold=cfg.patch_diff_threshold,
        keyframe_interval=cfg.keyframe_interval,
        smooth_fusion_enabled=cfg.smooth_fusion_enabled,
        fusion_mode=cfg.fusion_mode,
        semantic_shallow_layer=cfg.semantic_shallow_layer,
        semantic_threshold=cfg.semantic_threshold,
        attention_layer_id=cfg.attention_layer_id,
        attention_top_k=cfg.attention_top_k,
        visualize_attention=cfg.visualize_attention,
        visualization_save_dir=cfg.visualization_save_dir,
        visualization_interval=cfg.visualization_interval,
        attention_mode=cfg.attention_mode,
        use_multi_layer=cfg.use_multi_layer,
    )

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)

    # Initialize local logging
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # Initialize Weights & Biases logging as well
    if cfg.use_wandb:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_id,
        )

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    # Determine task range to evaluate
    task_start = cfg.task_start_id
    if cfg.num_tasks is not None:
        task_end = min(task_start + cfg.num_tasks, num_tasks_in_suite)
    else:
        task_end = num_tasks_in_suite
    
    # Validate task range
    if task_start >= num_tasks_in_suite:
        print(f"ERROR: task_start_id {task_start} exceeds available tasks {num_tasks_in_suite}")
        return
    if task_end > num_tasks_in_suite:
        print(f"WARNING: Requested task range exceeds available tasks. Adjusting to {num_tasks_in_suite}")
        task_end = num_tasks_in_suite
    
    num_tasks_to_run = task_end - task_start
    print(f"Running tasks {task_start} to {task_end-1} ({num_tasks_to_run} tasks total)")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    # Lists for performance metrics
    end_to_end_wall_times_ms = []
    end_to_end_cuda_times_ms = []

    for task_id in tqdm.tqdm(range(task_start, task_end)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            manager.reset_state()  # Reset AP-TUBE state for each new episode
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")

            # Reset environment
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            if cfg.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif cfg.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif cfg.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif cfg.task_suite_name == "libero_10":
                max_steps = 520  # longest training demo has 505 steps
            elif cfg.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")
            while t < max_steps + cfg.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    # Get preprocessed image
                    img = get_libero_image(obs, resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Prepare observations dict
                    # Note: OpenVLA does not take proprio state as input
                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }

                    # Query model to get action
                    # === Start E2E Timer ===
                    e2e_start_time = time.perf_counter()
                    e2e_start_event = torch.cuda.Event(enable_timing=True)
                    e2e_end_event = torch.cuda.Event(enable_timing=True)
                    e2e_start_event.record()

                    # Create visualization save directory for this episode if needed
                    vis_save_dir = None
                    if manager.visualize_attention and manager.fusion_mode in ["attention", "hybrid"]:
                        vis_save_dir = os.path.join(manager.visualization_save_dir, f"episode_{episode_idx}")
                    
                    action = get_action(
                        cfg,
                        model,
                        observation,
                        task_description,
                        processor=processor,
                        visualize_attention=manager.visualize_attention and manager.fusion_mode in ["attention", "hybrid"],
                        save_dir=vis_save_dir,
                        step_count=t,
                        visualization_interval=manager.visualization_interval
                    )

                    e2e_end_event.record()
                    torch.cuda.synchronize()
                    e2e_wall_time_ms = (time.perf_counter() - e2e_start_time) * 1000
                    e2e_cuda_time_ms = e2e_start_event.elapsed_time(e2e_end_event)

                    # Always collect performance metrics
                    end_to_end_wall_times_ms.append(e2e_wall_time_ms)
                    end_to_end_cuda_times_ms.append(e2e_cuda_time_ms)
                    # === End E2E Timer ===

                    # Normalize gripper action [0,1] -> [-1,+1] because the environment expects the latter
                    action = normalize_gripper_action(action, binarize=True)

                    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
                    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    print(f"Caught exception: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file
            )

            # Log current results
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # Log final results
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        if cfg.use_wandb:
            wandb.log(
                {
                    f"success_rate/{task_description}": float(task_successes) / float(task_episodes),
                    f"num_episodes/{task_description}": task_episodes,
                }
            )

    print(f"\nFinished evaluation for {num_tasks_to_run} tasks.")
    log_file.write(f"\nFinished evaluation for {num_tasks_to_run} tasks.\n")
    print(f"Overall success rate: {total_successes / total_episodes:.3f}")
    log_file.write(f"Overall success rate: {total_successes / total_episodes:.3f}\n")

    # Final Performance Report
    if end_to_end_cuda_times_ms:
        report_title = "AP-TUBE Performance Report" if manager.is_enabled() else "Baseline Performance Report"
        print("\n" + "=" * 50)
        print(report_title)
        print("=" * 50)

        # E2E Metrics
        e2e_wall_avg = np.mean(end_to_end_wall_times_ms)
        e2e_wall_std = np.std(end_to_end_wall_times_ms)
        e2e_wall_total = np.sum(end_to_end_wall_times_ms)
        e2e_cuda_avg = np.mean(end_to_end_cuda_times_ms)
        e2e_cuda_std = np.std(end_to_end_cuda_times_ms)
        e2e_cuda_total = np.sum(end_to_end_cuda_times_ms)

        # Vision Metrics from Manager
        vision_metrics = manager.metrics
        vision_wall_avg = np.mean(vision_metrics["vision_wall_time_ms"]) if vision_metrics["vision_wall_time_ms"] else 0
        vision_wall_std = np.std(vision_metrics["vision_wall_time_ms"]) if vision_metrics["vision_wall_time_ms"] else 0
        vision_wall_total = np.sum(vision_metrics["vision_wall_time_ms"]) if vision_metrics["vision_wall_time_ms"] else 0
        vision_cuda_avg = np.mean(vision_metrics["vision_cuda_time_ms"]) if vision_metrics["vision_cuda_time_ms"] else 0
        vision_cuda_std = np.std(vision_metrics["vision_cuda_time_ms"]) if vision_metrics["vision_cuda_time_ms"] else 0
        vision_cuda_total = np.sum(vision_metrics["vision_cuda_time_ms"]) if vision_metrics["vision_cuda_time_ms"] else 0


        # FLOPs & Reuse Metrics
        vision_flops_avg = np.mean(vision_metrics["vision_gflops"]) if vision_metrics["vision_gflops"] else 0
        reused_dino_avg = np.mean(vision_metrics["reused_patches_dino"]) if vision_metrics["reused_patches_dino"] else 0
        reused_siglip_avg = np.mean(vision_metrics["reused_patches_siglip"]) if vision_metrics["reused_patches_siglip"] else 0
        reused_total_slots_avg = np.mean(vision_metrics["reused_patches_total"]) if vision_metrics["reused_patches_total"] else 0
        
        total_patches_per_backbone = 256
        total_slots = 512

        reuse_ratio_dino = (reused_dino_avg / total_patches_per_backbone) * 100 if total_patches_per_backbone > 0 else 0
        reuse_ratio_siglip = (reused_siglip_avg / total_patches_per_backbone) * 100 if total_patches_per_backbone > 0 else 0
        reuse_ratio_total = (reused_total_slots_avg / total_slots) * 100 if total_slots > 0 else 0

        # Calculate total GFLOPs
        total_vision_baseline_gflops = cfg.baseline_dino_gflops + cfg.baseline_siglip_gflops
        baseline_e2e_gflops = cfg.baseline_e2e_gflops
        # We need to calculate what the non-vision part of the e2e GFLOPs is.
        # This is a bit of a hack since we don't have the exact number, but it's a reasonable estimation.
        # Original total vision GFLOPs was 368.9919
        non_vision_gflops = baseline_e2e_gflops - total_vision_baseline_gflops 
        e2e_gflops = non_vision_gflops + vision_flops_avg


        """
        baseline_dino_gflops: float = 158.0496  # Baseline GFLOPs for DINOv2
        baseline_siglip_gflops: float = 210.9423  # Baseline GFLOPs for SigLIP
        baseline_vision_total_gflops: float = 368.9919  # Baseline GFLOPs for total vision
        baseline_e2e_gflops: float = 3894.2080  # Baseline GFLOPs for end-to-end
        """
        report_str = (
            f"\n--- End-to-End Performance ---\n"
            f"Avg. Wall Time: {e2e_wall_avg:.2f} ± {e2e_wall_std:.2f} ms\n"
            f"Total Wall Time: {e2e_wall_total:.2f} ms\n"
            f"Avg. CUDA Time: {e2e_cuda_avg:.2f} ± {e2e_cuda_std:.2f} ms\n"
            f"Total CUDA Time: {e2e_cuda_total:.2f} ms\n"
            f"Estimated GFLOPs: {e2e_gflops:.4f} (Baseline: {baseline_e2e_gflops:.4f})\n"
            f"\n--- Vision Backbone Performance ---\n"
            f"Avg. Wall Time: {vision_wall_avg:.2f} ± {vision_wall_std:.2f} ms\n"
            f"Total Wall Time: {vision_wall_total:.2f} ms\n"
            f"Avg. CUDA Time: {vision_cuda_avg:.2f} ± {vision_cuda_std:.2f} ms\n"
            f"Total CUDA Time: {vision_cuda_total:.2f} ms\n"
            f"Estimated GFLOPs: {vision_flops_avg:.4f} (Baseline: {total_vision_baseline_gflops:.4f})\n"
            f"\n--- AP-TUBE Efficiency ---\n"
            f"Avg. Reused Patches (DINO):   {reused_dino_avg:.2f} / {total_patches_per_backbone} ({reuse_ratio_dino:.2f}%)\n"
            f"Avg. Reused Patches (SigLIP): {reused_siglip_avg:.2f} / {total_patches_per_backbone} ({reuse_ratio_siglip:.2f}%)\n"
            f"Avg. Reused Slots (Total):   {reused_total_slots_avg:.2f} / {total_slots} ({reuse_ratio_total:.2f}%)\n"
        )

        print(report_str)
        log_file.write("\n" + report_str)

        if cfg.use_wandb:
            wandb.log({
                "perf/e2e_wall_time_ms_avg": e2e_wall_avg,
                "perf/e2e_wall_time_ms_total": e2e_wall_total,
                "perf/e2e_cuda_time_ms_avg": e2e_cuda_avg,
                "perf/e2e_cuda_time_ms_total": e2e_cuda_total,
                "perf/e2e_gflops_avg": e2e_gflops,
                "perf/vision_wall_time_ms_avg": vision_wall_avg,
                "perf/vision_wall_time_ms_total": vision_wall_total,
                "perf/vision_cuda_time_ms_avg": vision_cuda_avg,
                "perf/vision_cuda_time_ms_total": vision_cuda_total,
                "perf/vision_gflops_avg": vision_flops_avg,
                "perf/reused_patches_dino_ratio_avg": reuse_ratio_dino,
                "perf/reused_patches_siglip_ratio_avg": reuse_ratio_siglip,
                "perf/reused_slots_total_ratio_avg": reuse_ratio_total,
            })

    if cfg.use_wandb:
        wandb.finish()

    log_file.close()


if __name__ == "__main__":
    eval_libero()
