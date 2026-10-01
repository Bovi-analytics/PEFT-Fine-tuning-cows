# Corrected test evaluation

Test results of the eight QLoRA/DoRA models re-computed with the training-time preprocessing (inputs rescaled to [0, 1] before DINOv3, fp16 autocast, base prepared as during training), using the notebooks in `notebooks/corrected_testing`. Started 1 October 2026 on NVIDIA A10 GPUs; the confusion matrices, per-crop predictions and a summary will be added here when the runs finish.
