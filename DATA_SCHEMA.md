# Data interface

`prepare_dataset.py` consumes a construction JSON with `occurrences`, `splits`, `survival`, `queries`, `correspondences`, `evolution_paths` and `feature_provenance`. All paths in that JSON are relative to its directory.

An occurrence has `occurrence_id`, `char_id`, `era`, `time`, `source`, `input_visibility` and either a 352D `.npy` feature path or an image plus optional feature `blocks`. `char_id` identifies the character population; `occurrence_id` identifies the individual variant. Galleries additionally have `gallery:true` and a candidate `character`. Era values are `OBI`, `Bronze`, `Seal`, `Clerical`, `Regular`.

`input_visibility` is `label_free` for query observations or `candidate_reference` for known reference metadata. Query features must not use the answer's definition or components. Feature provenance records extraction/version, projection weights and fitting population.

## Supervision

All three partitions (`train`, `val`, `test`) are character-disjoint.

- `splits` contains `{pair_id, source_id, target_id, pair_type}` records. `pair_type` is `adjacent`, `skip` or `complete`. Complete records additionally supply `complete_chain_ids` with one real occurrence for every era.
- `survival` contains `{source_id, target_time, label, evidence}` records with OBI sources and independently supported binary outcomes. Corpus absence alone is insufficient to label extinction.
- `queries` contains `{query_id, source_id, ground_truth}` records. `ground_truth` may be null for unresolved observations.
- `correspondences` maps a source occurrence ID to target-era occurrence lists: `{source_id: {Seal: [target_id]}}`. Evaluation query-to-answer links are excluded.
- `evolution_paths` maps each candidate character to `{bronze_id, source}` records. All referenced Bronze occurrences must exist.

## Modality blocks

Image extraction accepts `formation_type` (0–5), `component_embedding` (42D), `layout` (16D), `semantic_embedding` (64D), `cooccurring_features` (other-occurrence 256D vectors), `context_projection` (256×64), `coordinates` ([latitude, longitude]) and `cnn_density` (16D). Array-valued blocks can be supplied directly or via relative `.npy` paths. Missing values are zero.

## Prepared output

The builder writes `metadata.json`, `features/`, portable `images/`, partitioned pair/survival/query JSON, `databases/*_database.pt`, `correspondences.json`, `evolution_paths.json` and the construction records. Features are finite float32 vectors. Gallery image pointers remain relative to the prepared root and are resolved when evaluating.

## CCAMC source format

`data/metadata/occurrences.jsonl.gz` contains one CCAMC occurrence per line, with original character, writing category, dynasty, detailed period, image URL, bibliography, object/collection reference, context and cached-page pointers. `image_path`, `cache_sources` and `source_book_cache` are relative to `data/raw/`. Empty image paths preserve occurrences without a linked local asset. `src/data/ccamc.py` streams this format. The six-category source corpus and five-era model supervision use distinct schemas.

## Step-wise paths and pruning

The `stepwise_modern` verifier takes `{candidate_character:[{source:"edition identifier",occurrences:{Regular:"r_id",Clerical:"c_id",Seal:"s_id",Bronze:"b_id"}}]}`. All active era occurrences must belong to that candidate and have strictly ordered times. A threshold file contains `thresholds`, validation query/path identifiers and a fixed quantile and the exact calibrated checkpoint SHA-256; every active reverse checkpoint and OBI must have a finite nonnegative cutoff.

## Source adapter

`CCAMCSourceDataset(data_root)` addresses all original occurrences. Unsupported categories and missing assets return `features:null`; `eligible_indices` explicitly identifies actual image-bearing OBI/Bronze items. `metadata/model_input_index.jsonl.gz` retains every occurrence ID, source dynasty/period and linked path. Exact recognized subperiod labels map to Table 1 centers; otherwise the model receives an explicit `era_center_default` time resolution. Unknown source categories have `era:null,time:null`. No source period is overwritten.

## External queries and predictions

`BenchmarkQueryDataset` accepts source-bearing JSON/JSONL records with `query_id`, `source_id`, `source`, `input_visibility:"label_free"`, `time`, optional `ground_truth`, and either a relative `features` path or `image`. Actual supplied feature provenance is required by the construction interface. Population scoring consumes `{query_id,ground_truth,attributes}` rows and predictions with ranked candidates or score maps. Missing predictions count as misses; null truths are kept outside accuracy denominators. A `trials` list contains actual separate trial predictions.

## Photo-choice manifest

The optional association adapter consumes a JSON list with `query_id`, `question_index` (the official quiz index), `char_id`, `split`, `source`, `input_visibility:"label_free"`, `time`, `features` or `query_image`, `options:[{key:"A",image:"photo.png"},...]` (exactly four unique keys), and `correct_answer`. Paths are relative to the manifest. Character labels are disjoint across train/val/test; the target key is supervision only. The final scorer consumes the original official quiz separately. Adapter checkpoints bind to the exact frozen encoder weights.

## Additional annotations

Radical/formation/scribal/error grouping consumes explicit occurrence or population attributes. Expert ratings require actual rater and case identifiers. Latent expressions specify real occurrence IDs and coefficients plus a fixed gallery. Shape manifests contain `char_id`, `occurrence_id`, `era`, `image`, `source` and optional independently annotated `stroke_count`. No semantic labels, expert ratings, later-era glyphs or survival outcomes are generated from a source image automatically.
