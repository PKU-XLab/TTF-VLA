# 🦾 TTF-VLA: Temporal Token Fusion for Vision-Language-Action Models

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2508.19257-b31b1b.svg)](https://arxiv.org/abs/2508.19257)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

A training-free inference optimization method for enhancing VLA model performance through intelligent temporal visual information integration.  

Project and repository led and maintained by [Chenghao Liu](https://github.com/MrCapricornLiu).

We are thrilled to announce that our paper has been accepted by **AAAI 2026** 🎉. Read it on [arXiv](https://arxiv.org/abs/2508.19257).

## 📦 Repository Structure

```bash
TTF-VLA/
├── experiments/robot/
│   ├── aptube_manager.py           # Core TTF implementation
│   ├── libero/
│   │   └── run_libero_eval_aptube.py  # Core evaluation script (LIBERO Env)
│   └── openvla_utils.py            # VLA model utilities
├── prismatic/extern/hf/
│   └── modeling_prismatic.py       # Model integration points
└── README.md                       # This file
```

> Note: **aptube** is the old name of TTF. To avoid unnecessary bugs, we kept it unchanged in the code.

## ⚙️ Setup

If you don't have mamba, run the following command first:

```bash
conda install -n base -c conda-forge mamba
mamba --version
mamba shell init --shell bash
exec $SHELL -l
```

Either mamba or conda works; mamba is much faster.
**If you prefer conda**, simply replace all `mamba` commands below with `conda`.

```bash
mamba create -n ttfvla python=3.10 -y
mamba activate ttfvla
mamba install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y  

# pwd: ~/TTF-VLA
# VLA-related packages
pip install -e .

# LIBERO Environment packages
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO 
# pwd: ~/TTF-VLA/LIBERO
pip install -e .

cd .. 
# pwd: ~/TTF-VLA
pip install -r experiments/robot/libero/libero_requirements.txt
```

⚠️ Afterward, this error may occur:
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. tensorflow 2.15.0 requires numpy<2.0.0,>=1.23.5, but you have numpy 2.2.6 which is incompatible.
```
Then just run 
```bash
pip install numpy==1.26
```
⚠️ After that, **it will report a conflict error between numpy and tensorflow, just ignore it.**

Because we need to extract attention during inference, we don't use flash attention. Don’t worry, this has almost no impact on inference speed for VLA models.

## 🧩 Download Pretrained Models

```bash
mamba activate ttfvla
python download-ckpt-scripts/download_model_local.py  --model_id openvla/openvla-7b-finetuned-libero-spatial
```
You can change the model_id to download other models as needed:
- `openvla/openvla-7b-finetuned-libero-object`
- `openvla/openvla-7b-finetuned-libero-goal`
- `openvla/openvla-7b-finetuned-libero-10`

Note that openvla-7b-finetuned-libero-10 is the name of model finetuned on LIBERO-Long task suite.

## 🧪 Evaluation on LIBERO Tasks

```bash
# Evaluate OpenVLA + TTF on Object task suite
# pwd: ~/TTF-VLA
python experiments/robot/libero/run_libero_eval_aptube.py \
    --pretrained_checkpoint "checkpoints/openvla-7b-finetuned-libero-object" \
    --task_suite_name "libero_object" \
    --center_crop True \
    --aptube_enabled True \
    --fusion_mode "attention_guided" \
    --keyframe_interval 3 \
    --patch_diff_threshold 0.03 \
    --attention_top_k 70 \
    --num_trials_per_task 20
```

**Available task suites:**

* 🧱 `libero_object` — Object manipulation tasks
* 📐 `libero_spatial` — Spatial reasoning tasks
* 🎯 `libero_goal` — Goal-conditioned tasks
* ⏱️ `libero_10` — Long-horizon tasks

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025ttf,
  title={TTF-VLA: Temporal Token Fusion via Pixel-Attention Integration for Vision-Language-Action Models},
  author={Liu, Chenghao and Zhang, Jiachen and Li, Chengxuan and Zhou, Zhimu and Wu, Shixin and Huang, Songfang and Duan, Huiling},
  journal={arXiv preprint arXiv:2508.19257},
  doi={10.48550/arXiv.2508.19257},
  url={https://arxiv.org/abs/2508.19257}
}
```

## 🙏 Acknowledgements

We build upon the excellent works of [OpenVLA](https://github.com/openvla/openvla) and [VLA-Cache](https://github.com/siyuhsu/vla-cache). We sincerely appreciate their great work.

