# Data splits

`splits/train.csv` (2,160 crops), `splits/val.csv` (540) and `splits/test.csv` (211,800) list every image crop used in the study. Each crop shows one animal, cut from a video frame with a bounding box, and carries that animal's behavior label at that time in the source dataset.

* **Training and validation:** 300 crops per behavior, reviewed by two annotators with veterinary training against explicit criteria (Supplementary Information of the paper). They were split 240/60 per class with scikit-learn `train_test_split(test_size=0.2, stratify=label, random_state=42)`.
* **Test:** all remaining labeled crops of the nine behaviors in the two sources. Labels are the source annotations and were not visually verified. `test_index` is the row order used for evaluation and in all prediction files.

## Columns

| Column | Description |
|---|---|
| `split` | `train`, `val` or `test` |
| `test_index` | 0–211,799, test rows only; order of evaluation and of `results/*/test_predictions.csv.gz` |
| `source` | `MmCows` or `PlayBehavior` |
| `behavior` | behavior name used in the paper |
| `source_label` | label in the source dataset (MmCows calls the eating behaviors *Feeding head down/up*) |
| `crop_file` | MmCows: crop file name `ID<cow>+<camera>+<unix time>_<hh-mm-ss>_<hash>.jpg`. PlayBehavior: `pb_<8-hex hash>.jpg` |
| `animal_id` | MmCows: `cow_1` … `cow_16` as in MmCows. PlayBehavior, released farms: `calf_id` of the public release (e.g. `TOL-2642`, `EEM-9`). PlayBehavior, unreleased farms: pseudonymous `unreleased_farm_k_calf_n` |
| `site` | MmCows: the single MmCows pen. PlayBehavior: farm code of the public release (`Tol`, `Eem`) or pseudonymous `unreleased_farm_1`, `unreleased_farm_2` |
| `camera` | MmCows: `cam_1` … `cam_4` (synchronized, so the same cow at the same second appears in several views). PlayBehavior: camera channel in the original file name |
| `timestamp_unix` | MmCows only: Unix time of the frame (1 frame per second) |
| `public_session_id` | PlayBehavior, released farms: session of the public release (e.g. `Tol2_2024-06-11_PM`) |
| `public_frame` | PlayBehavior, released farms: frame of that session (e.g. `0135351.jpg`) |
| `public_frame_tar` | PlayBehavior, released farms: shard of the public release that contains the frame |

Class order used by all prediction and confusion-matrix files (index 0–8): Drinking, Eating head down, Eating head up, Lying, Standing, Walking, Frontal pushing, Gallop, Leap.

## Finding the source frames

* **MmCows** rows keep the identifiers of the MmCows release (cow, camera, Unix time). The crops can be regenerated from its frames and bounding-box annotations.
* **PlayBehavior.** 2,528 of the 2,819 calf crops come from the two farms whose video, decoded frames, annotations and calf tracks are public under CC0 ([Sonam5/Calf-Play-Behavior-Dataset](https://huggingface.co/datasets/Sonam5/Calf-Play-Behavior-Dataset), doi:10.57967/hf/10695). Each of these rows gives the session, frame, shard and calf of the public release. Every frame was checked against the release's frame index.

  The frame can be read directly:

  ```python
  import io, tarfile
  from PIL import Image
  from huggingface_hub import hf_hub_download
  row = {"public_frame_tar": "frames/Tol2_2024-06-11_PM/Tol2_2024-06-11_PM_video046.tar", "public_frame": "0135351.jpg"}
  path = hf_hub_download("Sonam5/Calf-Play-Behavior-Dataset", row["public_frame_tar"], repo_type="dataset")
  with tarfile.open(path) as tf:
      frame = Image.open(io.BytesIO(tf.extractfile(row["public_frame"]).read()))
  ```

  The release's `tracking/` files give automatic SAMURAI boxes for most tracked calves. The boxes used to cut the crops in this study are not part of this repository.
* **Unreleased farms.** The other 291 crops come from two farms without an agreement for open release. They are pseudonymized: farm, calf, date and time are not given. They are available from the corresponding author on reasonable request.

## Licences and sources

* **MmCows.** Vu, H. et al. MmCows: a multimodal dataset for dairy cattle monitoring. *Adv. Neural Inf. Process. Syst.* 37 (2024); <https://github.com/neis-lab/mmcows>.
  * Released under CC BY-NC-SA 4.0 (Hugging Face) / CC BY-NC 4.0 (paper).
  * Recorded with the approval of the University of Wisconsin–Madison IACUC (protocol A006606).
  * Rows derived from MmCows are shared under CC BY-NC-SA 4.0.
* **PlayBehavior.** Yang, H., Lesscher, H., Liu, E. & Hostens, M. arXiv:2602.00111 (2026).
  * Public recordings: Yang, H., Lesscher, H., Liu, E. & Hostens, M. *Video recordings and play behaviour annotations of group-housed dairy calves on two Dutch farms*. Hugging Face, doi:10.57967/hf/10695 (2026), CC0 1.0.
  * Recorded with the farmers' consent and the approval of the Animal Welfare Body of Utrecht University (approval 10818-2024-02).
