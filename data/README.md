# CCAMC fine-grained period corpus

The corpus contains 158,620 source occurrences across six original writing categories. Records preserve dynasty and detailed-period labels, glyph URLs, edition/object references, context and source-page locations. Bibliographical expansions are taken from the cached CCAMC reference tables and retain their matching abbreviation and page.

`metadata/occurrences.jsonl.gz` is the canonical row-oriented export; `occurrences.csv.gz` is its tabular representation. `dataset_statistics.json` provides reproducible descriptive counts. `manifest.json` identifies assets and relative paths. Images and source pages are in `raw/`; only actual files referenced by this corpus are included. Empty image paths remain in the population.

A row is an occurrence, not an independent character, object or source image. Several occurrences can share an image URL or local image. The source's six writing categories and detailed periods are preserved without relabeling them as model era correspondences or survival outcomes.

Source: CCAMC ancient and modern Chinese character collection, http://ccamc.org/cjkv_oaccgd.php . The included source files retain their original source rights.

## Language and category labels

All release documentation and presentation labels are in English. Original characters, dynasty/period values, scribal-group names, bibliography, URLs and cached source pages retain their source language for verification. `metadata/category_labels.json` maps the six original category values to their English display labels. The original categorical values are data, rather than explanatory prose. `metadata/model_input_index.jsonl.gz` provides the occurrence-preserving model interface.
