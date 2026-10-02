# Parameter-efficient fine-tuning of a billion-parameter vision model improves dairy cattle behavior classification under a 1:98 train-to-test ratio

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%E2%80%932.6-red.svg)](https://pytorch.org/)
[![Code licence: MIT](https://img.shields.io/badge/code-MIT-yellow.svg)](LICENSE)
[![Data licence: CC BY-NC-SA 4.0](https://img.shields.io/badge/data-CC%20BY--NC--SA%204.0-lightgrey.svg)](data/README.md)
[![arXiv](https://img.shields.io/badge/arXiv-2603.17782-b31b1b.svg)](https://arxiv.org/abs/2603.17782)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-adapters-yellow)](https://huggingface.co/collections/Sonam5/peft4cows)

Haiyu Yang, Sumit Sharma, Enhong Liu and Miel Hostens — Department of Animal Science, Cornell University

Code, data splits, per-crop predictions and adapters for a controlled comparison of how a 6.7-billion-parameter self-supervised vision model (DINOv3 ViT-7B/16) can be adapted to nine dairy cattle behaviors from 2,160 expert-verified images on a single 16 GB GPU: networks trained from scratch (ResNet-18, ViT-Small), a frozen backbone with a trained head, and eight configurations of quantized low-rank adaptation (QLoRA and DoRA; ranks 8, 16, 64; adapters on the query projections or on all linear layers). All models are evaluated on 211,800 crops from the same video sources, 98 times the training set.

> **Note on the test evaluation (October 2026).** During training and validation, the DINOv3 model wrapper converted the ImageNet-normalized input back to pixel values in [0, 1] before the backbone. The original test-evaluation cells (Module 11) of the eight QLoRA/DoRA notebooks passed the normalized tensor directly. The DoRA cells also ran without fp16 autocast, and the two QLoRA q_proj notebooks merged the adapters into the 4-bit weights before testing. The frozen-backbone and from-scratch baselines were evaluated consistently.
>
> Two configurations were re-evaluated with the training-time preprocessing on the same random sample of 2,000 test crops ([`results/preprocessing_check`](results/preprocessing_check)). Of their predictions, 2.8% (QLoRA, all-linear, r = 64) and 4.0% (DoRA, q_proj, r = 8) changed, and accuracy changed by +0.35 and +0.10 percentage points (McNemar p = 0.34 and 0.89). Re-running the original procedure reproduced 99.9% of the original predictions. The other six configurations were not re-evaluated.
>
> **The results reported in the paper, in arXiv v1 (2603.17782) and in [`results/original_evaluation`](results/original_evaluation) are those of the original evaluation.** The evaluation cells in `notebooks/` have been corrected; changed lines are marked `FIX` and a note sits above each cell. The stored cell outputs are those of the original run. [`notebooks/corrected_testing`](notebooks/corrected_testing) contains stand-alone notebooks that evaluate the test set either as validation was evaluated (`preprocessing = corrected`) or as in the original evaluation (`legacy`).
>
> Separately, the three baselines were evaluated with their last-epoch weights rather than their best-validation weights (a shallow `state_dict().copy()` in their training loop); this is documented in the paper.

## Repository structure

```
PEFT-Fine-tuning-cows/
├── data/
│   ├── README.md                    column definitions, licences, anonymization
│   └── splits/{train,val,test}.csv  2,160 / 540 / 211,800 crops
├── notebooks/
│   ├── Q-lora ... / DoRA ... .ipynb training and evaluation of the eight adapted models (evaluation cell corrected, see above)
│   ├── TrainFromScatch_Preprocesing Pipeline Val = 1.ipynb    ResNet-18 and ViT-Small from scratch
│   ├── UsingPretrainedModel_DinoV3 Embeddings Val = 1.ipynb    frozen DINOv3 backbone + MLP head
│   └── corrected_testing/           Databricks notebooks that re-evaluate the adapted models (corrected or original preprocessing)
├── results/
│   ├── original_evaluation/         confusion matrices, per-crop predictions, summary (the results reported in the paper)
│   ├── preprocessing_check/         re-evaluation of two adapted models on 2,000 test crops (see the note above)
│   └── training_curves/             validation accuracy and training loss per epoch for all 11 runs
├── analysis/compute_metrics.py      recomputes every reported metric from the confusion matrices
├── assets/
├── environment.yml, requirements.txt, LICENSE
```

## Data

| Source | Behaviors | Licence | Access |
|---|---|---|---|
| [MmCows](https://github.com/neis-lab/mmcows) (Vu et al., NeurIPS 2024 Datasets and Benchmarks) | drinking, eating head down, eating head up, lying, standing, walking | CC BY-NC-SA 4.0 ([Hugging Face release](https://huggingface.co/datasets/neis-lab/mmcows)) | public |
| PlayBehavior (Yang et al., arXiv:2602.00111) | frontal pushing, gallop, leap | CC0 1.0 ([Hugging Face release](https://huggingface.co/datasets/Sonam5/Calf-Play-Behavior-Dataset), doi:10.57967/hf/10695) | public for the two farms that supplied 2,528 of the 2,819 calf crops; the 291 crops from two other farms on reasonable request |

`data/splits` lists every crop with its split, behavior, source label, animal and camera. MmCows rows give the Unix timestamp, so the crops can be regenerated from the public frames and bounding boxes. PlayBehavior rows from the two released farms give the farm, calf, session, frame and shard of the public calf dataset. Rows from the two unreleased farms are pseudonymized (`unreleased_farm_k`, `unreleased_farm_k_calf_n`; no dates or times). The split is the stratified 80/20 split of the 2,700 verified crops with `random_state=42` used in all notebooks. See [`data/README.md`](data/README.md).

## Reproducing the results

1. Environment: `requirements.txt` / `environment.yml` (the paper's runs used Databricks GPU runtimes with PyTorch 2.3.1 or 2.6.0, Transformers 4.57.1, PEFT 0.13.2, bitsandbytes 0.48.0, on one Tesla V100-PCIE-16GB).
2. Training: the notebooks in `notebooks/` (Module 0 builds the splits; Modules 5–10 train; Module 11 evaluates).
3. Re-evaluation: `notebooks/corrected_testing/Corrected testing - <model>.py` (Databricks source format). Each loads the saved best-validation adapters and head. It evaluates the 211,800 test crops, or a seeded random subset (widget `limit`), as validation was evaluated during training. Widget `preprocessing = legacy` reproduces the original evaluation for comparison. A full run is estimated at 8–18 h per model on one A10 GPU.
4. Metrics: `python analysis/compute_metrics.py results/original_evaluation/confusion_matrices.json` prints accuracy (with Wilson 95% intervals), weighted and macro-F1, balanced accuracy, per-class precision/recall/F1 and per-source accuracy.

## Trained models

Adapters and classification heads of the best QLoRA and DoRA configurations (all-linear, r = 64), the frozen-backbone head and the from-scratch networks are on Hugging Face: <https://huggingface.co/collections/Sonam5/peft4cows>. The backbone `facebook/dinov3-vit7b16-pretrain-lvd1689m` is gated and must be downloaded under its own licence. **The adapters expect 224 × 224 RGB inputs with pixel values in [0, 1] (no ImageNet normalization)**; each model card includes a working inference example.

## Citation

```bibtex
@article{yang2026peftcows,
  title   = {Parameter-efficient fine-tuning of a billion-parameter vision model improves dairy cattle behavior classification under a 1:98 train-to-test ratio},
  author  = {Yang, Haiyu and Sharma, Sumit and Liu, Enhong and Hostens, Miel},
  journal = {arXiv preprint arXiv:2603.17782},
  year    = {2026}
}
```

## Licence

Code: MIT (see `LICENSE`). Files in `data/` and `results/` that describe MmCows crops are derived from MmCows and are shared under CC BY-NC-SA 4.0; please also cite the MmCows paper when using them.

![Graphical summary of arXiv v1](./assets/Infographics.png)

*Graphical summary of arXiv v1 (generated with NotebookLM).*
