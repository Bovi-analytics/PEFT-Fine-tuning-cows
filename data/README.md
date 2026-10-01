# Data splits

`splits/train.csv` (2,160 crops), `splits/val.csv` (540) and `splits/test.csv` (211,800) list every image crop used in the study. Each crop shows one animal, cut from a video frame with a bounding box, and carries the behavior label of that animal at that time in the source dataset.

* **Training and validation:** 300 crops per behavior, reviewed by two annotators with veterinary training against explicit criteria (Supplementary Table S7 of the paper), split 240/60 per class with scikit-learn `train_test_split(test_size=0.2, stratify=label, random_state=42)`.
* **Test:** all remaining labeled crops of the nine behaviors in the two sources; labels are the source annotations, not visually verified. `test_index` is the row order used for evaluation and in all prediction files.

## Columns

| Column | Description |
|---|---|
| `split` | `train`, `val` or `test` |
| `test_index` | 0–211,799, test rows only; order of evaluation and of `results/*/test_predictions.csv.gz` |
| `source` | `MmCows` or `PlayBehavior` |
| `behavior` | behavior name used in the paper |
| `source_label` | label in the source dataset (MmCows calls the eating behaviors *Feeding head down/up*) |
| `crop_file` | MmCows: crop file name `ID<cow>+<camera>+<unix time>_<hh-mm-ss>_<hash>.jpg`; PlayBehavior: pseudonymous `pb_<random hash>.jpg` |
| `animal_id` | MmCows: `cow_1` … `cow_16` as in MmCows; PlayBehavior: pseudonymous `site_k_calf_n` |
| `site` | MmCows: the single MmCows pen; PlayBehavior: pseudonymous recording site `site_1` … `site_5` |
| `camera` | MmCows `cam_1` … `cam_4` (synchronized; the same cow at the same second appears in several views); PlayBehavior camera channel |
| `timestamp_unix` | MmCows only: Unix time of the frame (1 frame per second) |

Class order used by all prediction and confusion-matrix files (index 0–8): Drinking, Eating head down, Eating head up, Lying, Standing, Walking, Frontal pushing, Gallop, Leap.

## Anonymization

MmCows rows keep the public identifiers of the MmCows release, so the crops can be regenerated from its frames and bounding-box annotations. PlayBehavior was recorded on commercial farms; its rows were pseudonymized before release: farm abbreviations, calf ear-tag numbers and coat descriptions, recording dates and times and the original file names were replaced by `site_k`, `site_k_calf_n` and random crop identifiers. The mapping is kept by the authors.

## Licences and sources

* **MmCows** — Vu, H. et al. MmCows: a multimodal dataset for dairy cattle monitoring. *Adv. Neural Inf. Process. Syst.* 37 (2024); <https://github.com/neis-lab/mmcows>; released under CC BY-NC-SA 4.0 (Hugging Face) / CC BY-NC 4.0 (paper). Recorded with the approval of the University of Wisconsin–Madison IACUC (protocol A006606). Rows derived from MmCows are shared under CC BY-NC-SA 4.0.
* **PlayBehavior** — Yang, H., Lesscher, H., Liu, E. & Hostens, M. arXiv:2602.00111 (2026). Recorded on 14 commercial farms in the Netherlands with the farmers' consent, in accordance with Directive 2010/63/EU and with approval of the Animal Welfare Body of Utrecht University. The crops are available from the corresponding author on reasonable request.
