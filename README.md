# MSEF and CBED

Implementation of the manifold-based script-evolution model and cascaded bidirectional retrieval. The code includes five-modality features, an era-conditioned encoder, forward/inverse Neural ODE dynamics, separate survival training, CBED inference and evaluation.

## Install

Python 3.9+ and PyTorch 2+.

```bash
pip install -r requirements.txt
```

SVG rasterization uses CairoSVG and the Cairo shared library. On Linux install `libcairo2`; on macOS install Cairo with Homebrew. Optional local BERT preprocessing uses `requirements-bert.txt`. Tested package versions are in `environment-tested.json`.

## CCAMC source data

The complete CCAMC source export is included in `data/`: 158,620 occurrence records, 41,931 glyph image files and 29,650 cached source pages. Inspect or reproduce its descriptive statistics:

```bash
python scripts/summarize_ccamc.py --records data/metadata/occurrences.jsonl.gz --output results/ccamc --csv
```

The supplied source data directly supports an OBI/Bronze preparation path:

```bash
python scripts/export_ccamc_model_index.py --records data/metadata/occurrences.jsonl.gz --output data/metadata/model_input_index.jsonl.gz
python scripts/prepare_ccamc_evolution.py --data_root data --output prepared_ccamc
python src/train.py --config configs/ccamc_partial.json --data_dir prepared_ccamc --output_dir runs/ccamc --seed 42
python scripts/evaluate_retrieval.py --checkpoint runs/ccamc/best_model.pt --data_dir prepared_ccamc --source_era OBI --target_era Bronze --output results/ccamc_retrieval
```

All original source rows remain in the source export and model index. Preparation selects up to four actual image variants per character/era and records the selected population; characters are split 70/10/20 with seed 42. Positive pairs use CCAMC's published head-character grouping. Five-era CBED requires separately supplied Seal, Clerical and Regular occurrences and verified lineage records. No extinction labels are inferred from corpus absence.

The source archive includes fine-grained periods, glyph images, bibliography and cached source pages. Its six source writing categories retain their original labels. The model's five-era supervision interface is specified in `DATA_SCHEMA.md`.

## Feature preparation

```bash
python scripts/fit_projection.py --index train_vectors.json --output projection --output_dim 64
python scripts/encode_definitions.py --records definitions.json --bert /path/local_bert --projection projection/projection.npy --output semantic_features
python scripts/prepare_dataset.py --annotations construction.json --output prepared_data
```

`fit_projection.py` fits an uncentered SVD projection on source-bearing training records. Supplied frozen 352D features can instead be retained using `--frozen_features`. SVG and transparent raster inputs are composited on white. Prepared images and feature paths are portable.

## Train

```bash
python src/train.py --config configs/default.json --data_dir prepared_data --output_dir runs/seed42 --seed 42 --device cpu
torchrun --standalone --nproc_per_node=8 src/train.py --config configs/default.json --data_dir prepared_data --output_dir runs/seed42_ddp --seed 42 --device cuda
```

The default schedule is 100 evolution epochs followed by 20 survival epochs, with independent validation selection. Set `train_survival:false` when independently supported survival labels are unavailable. `last.pt` supports resumption in the original run directory with the same configuration, population and world size; `best_model.pt` combines the selected stages.

## Evaluate

```bash
python src/evaluate.py --checkpoint runs/seed42/best_model.pt --data_dir prepared_data --database_dir prepared_data/databases --known_correspondences prepared_data/correspondences.json --evolution_paths prepared_data/evolution_paths.json --output_dir results/seed42
```

Outputs include per-query ranked candidates, scores and log-scores, visual/source pointers, path evidence and population-level metrics. All labeled queries remain in the evaluation denominator. Five-run aggregation is available in `scripts/summarize_runs.py`. Four-choice association predictions are scored separately by `scripts/evaluate_pictobi.py`.

## Verification

```bash
python -m pytest -q tests
```

The test suite exercises ODE gradients and inversion, objective reductions, CBED ranking, SVG loading, portable preparation, two-stage training, checkpoints and two-rank CPU execution. Method details are in `IMPLEMENTATION.md`. This release supplies implementation and CCAMC source-data statistics; paper benchmark scores require trained checkpoints and their corresponding evaluation populations.

## Experiment catalogue

`experiments.json` links each experiment family to its code, required inputs and protocol. `EXPERIMENTS.md` describes command usage and the distinctions between algorithm versions.

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py --id train_dimension_256 --inputs paths.json
python scripts/run_experiment.py --id train_dimension_256 --inputs paths.json --execute
```

The resolver prints the concrete command before execution. A training input file can be `{"prepared_data":"/path/to/prepared","output":"/path/to/runs"}`. Analysis entries accept their explicit arguments after `--`; inspect the respective script's `--help`.

## Data and evaluation outputs

The source export preserves glyph identities, fine-grained periods and bibliographic provenance. Benchmark tables are generated from source-backed evaluation populations and actual checkpoint predictions; external baseline and expert comparisons consume actual predictions and ratings.
