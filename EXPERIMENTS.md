# Experiment protocols

| Family | Entry point | Outputs |
|---|---|---|
| Main FGCCES, HUST-OBS, EVOBC | `src/evaluate.py` | All-query predictions, R@K, R@1%, MRR, source/path evidence |
| Five independent runs | `scripts/summarize_runs.py` | Mean, sample standard deviation, observed run count |
| Loss, static, time, feature, era, dimension, scale ablations | `configs/` + `src/train.py` | Selected checkpoints, losses, exact training populations |
| Cascade/backward/survival/pruning variants | `src/evaluate.py` flags | Same-population metrics and pruning traces |
| Threshold fitting | `scripts/fit_pruning_thresholds.py` | Validation-only distance quantiles and source identifiers |
| Cross-era retrieval | `scripts/evaluate_retrieval.py` | Character rankings from actual occurrences and individual times |
| Diffusion PF-ODE | `scripts/train_probability_flow.py` | Validation-selected conditional score checkpoint |
| PictOBI | `scripts/pictobi_association.py`, `evaluate_pictobi.py` | Source-backed adapter predictions and official four-choice scores |
| Multi-round, calibration, scribal/error strata, false-positive pruning | `scripts/score_predictions.py` | Frozen-population aggregate and per-query records |
| Expert evaluation | `scripts/summarize_expert_ratings.py` | Counts and observed ratings by criterion/method |
| Query-level paired comparison | `scripts/compare_predictions.py` | Exact discordant-pair test and paired query statistics |
| Seven-feature evolution clusters | `scripts/analyze_evolution_shapes.py` | Raw/missing trajectories, three cluster assignments and agreement |
| Manifold and velocity | `scripts/analyze_manifold.py` | Coordinates, composition/inversion, local Jacobian and grouping statistics |
| Attention/activation/PI | `scripts/analyze_representations.py` | Attention, head effects, layer-token effects and dimension interventions |
| Image saliency and occlusion | `scripts/visual_occlusion.py` | Patch sensitivity and equal-ink image interventions |
| Latent arithmetic | `scripts/latent_arithmetic.py` | Explicit expressions and reference-gallery rankings |
| Manifold neighborhood plots | `scripts/project_manifold.py` | Source-ID-preserving PCA or t-SNE coordinates |
| Efficiency | `scripts/benchmark_efficiency.py` | Actual parameters, repeated synchronized encoder/flow timings, GPU memory |
| Source corpus statistics | `scripts/summarize_ccamc.py` | Actual coverage and counts with distinct counting units |

Default independent seeds are 42–46. Compare the same query IDs, gallery, visibility and scoring protocol. Pair-count variants request an actual eligible population of that size and fail if it is unavailable. Era-count runs use matching training and inference variants. Complete-only uses actual complete supervision; an endpoint-only loss variant is separate.

Example analyses:

```bash
python scripts/fit_pruning_thresholds.py --help
python scripts/analyze_manifold.py --checkpoint RUN/best_model.pt --data_dir DATA --output_dir OUT --volume_samples 4
python scripts/analyze_representations.py --checkpoint RUN/best_model.pt --data_dir DATA --output_dir OUT --head_ablation
python scripts/score_predictions.py --population population.jsonl --predictions predictions.jsonl --output metrics.json
```

Official external populations and predictions must be provided for external baselines. External baseline comparisons use supplied predictions aligned to the same frozen query population. Expert evaluation uses independent ratings. The data package contains descriptive corpus statistics; benchmark metrics are computed from trained checkpoints and evaluation predictions.
