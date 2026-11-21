# Big Data Assignment 2 · GitHub User Activity Prediction

This project uses a two-layer graph neural network to predict six GitHub activities for each user on the official test list.

## Data Source

The raw GitHub event data used in this project comes from [GH Archive](https://www.gharchive.org/). We thank GH Archive for making the hourly event data publicly available.

## Environment Setup

The commands below use Bash on Linux with Python 3.11 and an NVIDIA GPU with a driver compatible with the PyTorch CUDA 12.1 build. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), then run the following from the root of this repository. Execute the remaining commands in the same Bash session.

```bash
uv sync --locked --python 3.11 --extra remote-cu121
source .venv/bin/activate
```

The remaining commands use `python` from this activated environment. When opening a new terminal, return to the project root and run `source .venv/bin/activate` again.

## Prepare the Data

The two snapshots use the same revised preprocessor and cohort from the [official Assignment 2 repository](https://github.com/nycubigdata-2026fall/BigData2026-Assignment2). Keep `cohort.txt` alongside `preprocess.py`. Run the commands from this repository's root.

| Snapshot | UTC hourly filenames (inclusive) | Parquet directory | Used for |
| --- | --- | --- | --- |
| Training | `2026-08-29-16.json.gz` through `2026-11-14-15.json.gz` | `./data/training/db` | Model training and checkpoint selection |
| Submission | `2026-08-29-16.json.gz` through `2026-11-21-3.json.gz` | `./data/inference/db` | Threshold calibration and submission inference |

The submission snapshot's 2,004 expected hourly files were present and passed gzip checks. Its coverage ends at 04:00 UTC on November 21 (12:00 Taiwan time), exclusive. For an exact training rerun, use the same source files as the original training snapshot.

If you already have the eight Parquet files for a snapshot, use the directory containing `Users.parquet` and the other seven tables. If you have raw GH Archive files, use the command for the snapshot you need:

```bash
# Training and checkpoint selection
python ./BigData2026-Assignment2/scripts/preprocess.py --raw ./data/gharchive-training --out ./data/training

# Threshold calibration and submission inference
python ./BigData2026-Assignment2/scripts/preprocess.py --raw ./data/gharchive --out ./data/inference
```

The preprocessor scans `--raw` recursively and writes the eight Parquet files under `--out/db`. Keep the hourly files in one shared archive if convenient. Point `--raw` directly at that archive when it contains exactly the hours for the desired snapshot. If it contains additional hours, point `--raw` at a directory of symbolic links to just the required files. The two example input paths can be directories of links into the same archive.

## Reproduce the Submission CSV

After [Environment Setup](#environment-setup), start here if you already have the submission snapshot's eight Parquet tables. Set `DB_DIR` to the directory containing `Users.parquet` and the other seven tables. If you only have hourly GH Archive files, first generate the **submission snapshot** in [Prepare the Data](#prepare-the-data).

The bundled checkpoint contains the trained model and feature encoder. The bundled JSON contains the six calibrated thresholds.

```bash
DB_DIR='./data/inference/db'
TEST_INPUT='./BigData2026-Assignment2/submission/test_input.csv'
EVALUATE_SCRIPT='./BigData2026-Assignment2/scripts/evaluate.py'
CHECKPOINT='./artifacts/submission/validation_checkpoint.pt'
THRESHOLDS='./artifacts/submission/thresholds.json'
DATA_THROUGH='2026-11-21T04:00:00Z'
PREDICTION_TIME='2026-11-21T04:00:00Z'
OUTPUT='output.csv'
```

```bash
python src/predict_test.py predict \
  --data-dir "$DB_DIR" \
  --data-through "$DATA_THROUGH" \
  --input-cutoff "$PREDICTION_TIME" \
  --test-input "$TEST_INPUT" \
  --checkpoint "$CHECKPOINT" \
  --thresholds "$THRESHOLDS" \
  --output "$OUTPUT"
```

| Argument | Meaning |
| --- | --- |
| `--data-dir` | Directory containing the eight preprocessed Parquet tables. |
| `--data-through` | Confirmed data coverage boundary in UTC. |
| `--input-cutoff` | Prediction time in UTC; input records must be available before this time. |
| `--test-input` | Official CSV listing the test users. |
| `--checkpoint` | Saved model and feature encoder. |
| `--thresholds` | JSON containing the six calibrated activity thresholds. |
| `--output` | Destination for the prediction CSV. |

Check the CSV format and user list with the official script:

```bash
python "$EVALUATE_SCRIPT" --pred "$OUTPUT" --input "$TEST_INPUT"
```

`--pred` specifies the prediction CSV and `--input` specifies the official test user list. The submission CSV passed the official format check for all 4,465 users. It contains six binary predictions per user. Choose a new output path when rerunning inference on a directory that already contains the CSV.

## Train from Scratch

### Train and select the checkpoint

Use the **training snapshot** from [Prepare the Data](#prepare-the-data). Set `TRAIN_DB_DIR` to the directory containing its eight Parquet files. The preset starts sampling on September 6 at 00:00 Taiwan time and uses the November 7–14 label window to select a checkpoint, ending at November 14 at 00:00 Taiwan time (November 13 at 16:00 UTC, exclusive).

```bash
TRAIN_DB_DIR='./data/training/db'
TRAIN_RUN='./artifacts/retrained'

python src/train_model.py --formal --data-dir "$TRAIN_DB_DIR" --output "$TRAIN_RUN"
```

`--formal` selects the formal training preset. Here, `--output` specifies a new directory for the checkpoint and training metrics.

### Calibrate the six thresholds

Use the **submission snapshot** from [Prepare the Data](#prepare-the-data). It supplies the November 14–21 calibration label window (Taiwan time) and is reused for submission inference. If the eight tables are stored elsewhere, adjust `DB_DIR` below.

```bash
DB_DIR='./data/inference/db'
DATA_THROUGH='2026-11-21T04:00:00Z'

python src/predict_test.py calibrate \
  --data-dir "$DB_DIR" \
  --data-through "$DATA_THROUGH" \
  --checkpoint "$TRAIN_RUN/validation_checkpoint.pt" \
  --output "$TRAIN_RUN/thresholds.json"
```

To use the newly trained model, set `CHECKPOINT="$TRAIN_RUN/validation_checkpoint.pt"` and `THRESHOLDS="$TRAIN_RUN/thresholds.json"`, then choose a new `OUTPUT` and run the inference and format-check commands above.
