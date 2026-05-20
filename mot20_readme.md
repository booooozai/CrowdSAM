# MOT20-01 for CrowdSAM

This note documents the local MOT20-01 adaptation flow. The current setup uses
MOT20-01 as a single-class pedestrian dataset and follows the same logging style
as the CrowdHuman runs.

## Data Layout

Expected files:

```text
dataset/mot20/
├── train/
│   └── MOT20-01/
│       ├── gt/gt.txt
│       ├── img1/*.jpg
│       └── seqinfo.ini
└── annotations/
    ├── mot20_01_10shot_train.json
    ├── mot20_01_10mini_val.json
    └── mot20_01_10val.json
```

Training always uses:

```text
dataset/mot20/annotations/mot20_01_10shot_train.json
```

The mini and full validation configs only differ in `data.json_file`.

## Commands

Use physical GPU cuda3 by prefixing commands with `CUDA_VISIBLE_DEVICES=3`.
Inside the scripts this GPU is exposed as local `cuda:0`, which matches the
existing `tools/test.py` and `tools/batch_eval.py` device handling.

Train the MOT20-01 10-shot adapter:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/train.py --config_file ./configs/mot20_01_mini.yaml
```

Run mini validation:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/batch_eval.py --config_file ./configs/mot20_01_mini.yaml -n 1
```

Run full validation:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/batch_eval.py --config_file ./configs/mot20_01.yaml -n 1
```

Visualize a small mini-val slice:

```bash
CUDA_VISIBLE_DEVICES=3 python tools/test.py --config_file ./configs/mot20_01_mini.yaml --visualize --start_idx 0 --end_idx 5
```

## Outputs and Logs

The adapter checkpoint is saved to:

```text
adapter_weights/mot20_01_10shot.pth
```

Mini validation writes to:

```text
outputs/mot20_01_mini_vis/
```

Full validation writes to:

```text
outputs/mot20_01_vis/
```

Logs follow the same layout as CrowdHuman:

```text
<output_dir>/log/<timestamp>.log
```

`tools/batch_eval.py` also writes:

```text
<output_dir>/record.txt
```

## Notes

- MOT20 mini/full validation uses `tools/crowdhuman_eval.py`, so batch logs
  report `AP`, `MR`, `Recall`, `tp`, and `fp`, matching the CrowdHuman-style
  metric layout. This is still bbox detection evaluation, not MOTChallenge
  tracking evaluation.
- The model is configured as a single-class pedestrian detector with
  `model.n_class: 1`.
- If image loading fails, check that each annotation `file_name` looks like
  `MOT20-01/img1/000002.jpg`; the project resolves it under
  `dataset/mot20/train/`.
