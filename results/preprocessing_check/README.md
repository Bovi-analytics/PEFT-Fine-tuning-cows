# Preprocessing check (October 2026)

During training and validation, the DINOv3 model wrapper rescaled the ImageNet-normalized crops back to [0, 1] before the backbone. The original test evaluation of the eight QLoRA/DoRA models passed the normalized crops directly (see the note in the main README). To estimate the effect, two of the eight models were re-evaluated on the same seeded random sample of 2,000 of the 211,800 test crops (`test_df.sample(n=2000, random_state=0)`) with the notebooks in [`notebooks/corrected_testing`](../../notebooks/corrected_testing), on one NVIDIA A10 GPU (24 GB; torch 2.7.0, transformers 4.57.1, PEFT 0.13.2, bitsandbytes 0.48.0):

* `legacy`: test crops ImageNet-normalized, as in the original evaluation (QLoRA with fp16 autocast, DoRA without, as in the original cells);
* `corrected`: test crops rescaled to [0, 1], fp16 autocast and the 4-bit base prepared with `prepare_model_for_kbit_training`, as in training and validation.

| Model | Mode | Accuracy (%) | Original evaluation, same crops (%) | Agreement with the original predictions (%) | Changed predictions |
|---|---|---|---|---|---|
| QLoRA, all-linear, r = 64 | legacy | 82.85 | 82.90 | 99.90 | 2 |
| QLoRA, all-linear, r = 64 | corrected | 83.25 | 82.90 | 97.20 | 56 |
| DoRA, q_proj, r = 8 | legacy | 82.30 | 82.25 | 99.95 | 1 |
| DoRA, q_proj, r = 8 | corrected | 82.35 | 82.25 | 96.05 | 79 |

* The legacy runs reproduce the original predictions. The one or two differing crops reflect GPU and library numerics (A10 with torch 2.7 versus V100 with torch 2.3/2.6). Data, crop order, adapters and heads are therefore loaded as in the original runs.
* With the training-time preprocessing, 2.8% and 4.0% of the predictions change. Accuracy on these crops changes by +0.35 and +0.10 percentage points. Of the changed predictions, 23 went from wrong to right and 16 from right to wrong for QLoRA (exact McNemar p = 0.34), and 29 and 27 for DoRA (p = 0.89). The rest changed from one wrong class to another.
* The sample was drawn from all test crops and holds only 19 PlayBehavior crops, so it says little about the three calf behaviors.
* The other six configurations were not re-evaluated. These include the two QLoRA q_proj models, whose original evaluation also merged the adapters into the 4-bit weights. The results reported in the paper are those of the original evaluation ([`results/original_evaluation`](../original_evaluation)).
* The two legacy runs ran at the same time on one GPU, at about 4.9 crops/s each, against about 7 crops/s for a single run.

## Files

| File | Content |
|---|---|
| `summary.csv` | the table above, with throughput, GPU memory and library versions |
| `check_predictions.csv` | per-crop predictions: `test_index` (row of [`data/splits/test.csv`](../../data/splits/test.csv)), `source`, `true` and, for each model, the `original`, `legacy` and `corrected` predictions (class indices 0–8 in the order given in [`data/README.md`](../../data/README.md)) |
| `<model>_<mode>.json` | full output of each run: confusion matrix, per-source accuracy, macro- and weighted F1, timing, library versions |
