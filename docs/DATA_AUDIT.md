# Source audit and release qualification

Audit date: 2026-09-22. Reference: the published Pattern Recognition article, DOI `10.1016/j.patcog.2026.113262`, and its local supplementary LaTeX source. Raw files are preserved.

## Recoverable source counts

| Platform | Source screenshots | Source relation groups | Paper relation samples |
|---|---:|---:|---:|
| Desktop | 181 | 1,518 | 6,854 |
| Mobile | 7,210 | 20,284 | 20,284 |
| Web | 917 | 10,586 | 10,586 |

Mobile and web match the paper exactly at the source-group level. Desktop does not use the same counting unit in the surviving artifacts:

- Full-screen desktop functional SFT file: 2,300 examples, including synthetic negatives.
- Desktop crop SFT files: 4,554 examples (1,518 groups × three relation dimensions).
- Desktop marked-crop SFT files: another 4,554 examples for the same groups and dimensions.
- Adding full-screen examples to **one** crop variant gives 6,854, but no recovered inclusion list identifies which variant was used.
- The surviving `get_relation_num` script adds both variants, producing 11,408 examples. It counts 9,108 crop instruction rows as images; this is not a count of unique source screenshots.

Matching the total alone does not establish the paper's historical dataset. Consequently, neither a crop variant nor a random 37,724-row subset is silently designated as the original release.

The combined `finetune_relation.json` contains 50,525 functional training examples: desktop 2,300, mobile 30,623, web 17,602. The mobile/web totals include synthetic negatives, so this file is not the 37,724-relation source corpus. An older local v1 preprocessing pass retained 45,900 examples; it also introduced a dev/test protocol absent from the paper. Those later artifacts are not reused as the published experimental split.

## Quality-filtered rebuild

The candidate uses canonical source groups with all available relation dimensions. Desktop crop augmentation and synthetic negatives are omitted because the historical inclusion list is unverified. This changes the training distribution and prevents claiming direct reproduction of published performance.

| Platform | Accepted groups | Benchmark-source groups | Training groups |
|---|---:|---:|---:|
| Desktop | 1,450 | 977 | 473 |
| Mobile | 20,273 | 512 | 19,761 |
| Web | 8,947 | 1,088 | 7,859 |
| **Total** | **30,670** | **2,577** | **28,093** |

Checks and changes:

1. All 7,210 mobile source files contain swapped width/height metadata. The rebuild uses decoded image dimensions and normalizes coordinates accordingly.
2. Reject 1,650 relation groups with invalid or out-of-image element boxes: 11 mobile and 1,639 web. Boxes are not silently clipped because clipping can change the target meaning.
3. Reject 66 desktop groups across files with ambiguous correspondence between element groups and relationship entries, plus two groups containing fewer than two elements.
4. Require nonempty element captions and functional descriptions. Preserve source descriptions and categorical labels without inventing corrected semantics.
5. Remove normalized exact duplicates if found. Use source paths and decoded RGB hashes to exclude RelationQA sources from every training task.
6. Record stable IDs, relative image paths, source file hashes, group indices, actual dimensions, and per-exclusion reasons.
7. Generate grounding boxes from source geometry instead of copying the legacy generator's incorrect `y2 / image_width` calculation.

The checks establish structural validity and split isolation. They do not constitute a new human review of every semantic relation or a near-duplicate visual similarity search.

## RelationQA

The original **1,009 questions** and labels are retained: desktop 587, mobile 200, web 222. Label counts are trigger 269, complement 260, parallel 230, none 250. They originate from **350 source screenshots**.

The benchmark is released as a single evaluation set, as described in the paper. The added local 202/807 dev/test split is not used. Gold labels are included for reproducible local scoring, and benchmark data must not be used for training.

The legacy `img_path` is an unmarked screenshot. The red-box screenshot is `line_img_path`, and `crop_img_path` is the crop. The rebuild maps these explicitly to `original`, `marked`, and `crop` and packages only the required views.

## Scope and remaining limitations

The historical desktop selection and UGround supplement are not recoverable from the inspected artifacts. GPU training, the paper's navigation-data conversion, and full model-result reproduction have not been validated. Public training defaults follow the paper, while the rebuilt relation training corpus explicitly includes three dimensions. This distinction is documented instead of claiming an exact historical code snapshot.

The original files contain screenshot content from external applications/websites and MobileViews. The build allowlist excludes crawler scripts, local credentials, logs, raw browser state and model weights. Publication does not create new rights to third-party screenshots. An open-source/data license has not been invented or assigned during this preparation.

The full machine-readable audit and individual exclusions are inside the candidate dataset package. At this preparation stage, dataset publication remains pending and the candidate is not advertised as an exact 37,724-sample reproduction.
