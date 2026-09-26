# Implementation specification

## Features

Input dimension is 352: visual 128, structural 64, semantic 64, context 64 and spatiotemporal 32. Unknown modalities use zero vectors. Character identity is used for split validation and candidate reference labels, and is excluded from model features.

The visual block has eight 16D groups: contour Fourier magnitudes, curvature histogram, gradient orientations, skeleton grid density, topology/bounding-box measurements, reflection/rotation similarities, ink density and a supplied frozen CNN density vector (zero when absent). Glyphs are rasterized on white, resized to 64×64 and binarized with Otsu thresholding. Structural features concatenate six formation indicators, a 42D component vector and 16D layout. Definition features use local BERT CLS vectors and a frozen 768×64 projection. Context averages projected visual/structural/semantic vectors of other occurrences from the same artifact. The projection utility fits an uncentered truncated SVD on training records and fixes component signs by the largest loading. Time has 16 sinusoidal coordinates; latitude and longitude supply eight coordinates each.

## Encoder and flow

Each of the five modality blocks is projected to 256D. A shared projection of all 352 inputs adds global context to each token. Four residual blocks use hidden width 512, GELU, LayerNorm and dropout 0.1. Twelve Transformer blocks use eight attention heads, a 1024D feed-forward layer and era-conditioned adaptive LayerNorm scale/shift. Mean token pooling and the output projection give the 256D manifold coordinate. A 64D sinusoidal time embedding conditions the encoder and dynamics.

The velocity field uses three spectral-normalized layers with hidden width 512. Forward and inverse dynamics integrate the same field with dopri5 and relative/absolute tolerances 1e-5. Identical time spans are solved together; zero-length spans return their input. Spectral estimates update once per training step and stay fixed within each solve.

Default era centers are OBI 0.05, Bronze 0.35, Seal 0.70, Clerical 0.85 and Regular 1.0. Supplied occurrence times are preserved. The declared script intervals are OBI [0,0.30), Bronze [0.30,0.70), Seal [0.70,0.85), Clerical [0.85,1), and Regular at 1. Scribal-group labels remain metadata and do not automatically establish chronological bins. The default forward cascade propagates the original query coordinate by successive ODE solves; it never re-encodes or re-anchors the query. Reference coordinates are explicitly transported from their recorded times to the retrieval checkpoint and cached.

## Optimization

Evolution minimizes squared Euclidean distances with weights adjacent 1.0, skip 0.5, complete endpoints 0.3 and roundtrip cycle 0.5. A complete-chain loss compares the OBI and Regular endpoints; the cycle reconstructs the actual forward prediction. Each component averages over its eligible population. Cycle supervision uses adjacent and skip pairs only (Appendix F defines P as their union); complete-chain records are not added to the pair denominator. Distributed gradients weight these populations globally.

The encoder and dynamics are then frozen. The survival network trains on independently evidenced OBI records with binary cross entropy scaled by 0.2; one denotes survival. The default schedule is 100 evolution and 20 survival epochs. Both stages use AdamW, learning rate 1e-4, weight decay 1e-4, global batch size 256 and clipping norm 1.0. Validation selects the checkpoint of each stage.

## CBED and metrics

The default `observed_checkpoint` implements final-manuscript Appendix G Algorithm 1. Survival is checked at Bronze and Seal only, from the original query encoding. Each check and its score are recorded. A low score yields `abstain` with `low_survival_score`; it is not a historical extinction claim. Retrieval takes five occurrences per era and unions the time-ordered reachable reference observations from all accumulated earlier candidate sets. Held-out query IDs and query-answer edges are forbidden.

Each permitted source-backed path starts from an independently observed Regular prototype. Inverse flow traverses the declared checkpoints, inserting the actual intermediate observation times when necessary. Distances are tested only where a prototype exists, always at matching times, and at the observed OBI query. Missing observations are skipped rather than zero-filled. A Regular-only path is permitted and explicitly marked `endpoint_only`. Each valid path is scored by its final OBI cosine, with the norm-product denominator floored at 1e-8. The candidate score is the maximum valid path cosine; scores remain raw compatibility values. Candidate identities are deduplicated and ties use sorted character IDs. No consistent candidate yields `abstain` with `no_consistent_candidate`. Unverified and pruned candidates remain recorded.

Every query produces a record. A labeled rejection, empty result or missing correct candidate is a retrieval miss. R@1% uses the full Regular character gallery, rather than the retained shortlist. Metrics include recall, reciprocal rank and average precision under one correct character per query. Confidence is a normalized score within the supplied candidate set.

## Version selection

Thresholds are fitted on validation positive lineages with `fit_pruning_thresholds.py`, using distance quantile 0.95. Calibration is bound to the exact checkpoint, configuration, seed, verifier version and validation query IDs. Missing-era paths contribute only their observed checkpoint distances; test queries cannot participate in fitting. Every checkpoint used for an actual comparison requires a fitted threshold. `--disable_pruning` is an explicit ablation.

`bronze_path_mean` and `stepwise_modern` retain earlier manuscript variants for explicit comparisons. The former averages Bronze inverse coordinates and uses exp(-distance); the latter requires complete paths and uses exp(cosine-1). Neither is the final-manuscript default. To use the historical re-encoding variant, select `--verification_mode bronze_path_mean --forward_mode projection`. Legacy calibration must explicitly select `--verification_mode stepwise_modern` in both fitting and evaluation.

The default architecture uses eight attention heads and 256D residual coordinates. `attention_144_heads.json` selects twelve heads per layer, twelve layers, a 384D internal attention projection and 256D residual coordinates. Parameter counts are computed from the instantiated model.

Default settings are five modality tokens, 20 survival epochs and pruning distance quantile 0.95. The conventional CNN photo adapter is configured separately. Each run records its resolved configuration and selected training population.

## Conditional diffusion baseline

The VP baseline learns a noise predictor by denoising score matching in a frozen, shared encoder space. Its noise-time schedule is beta(u)=0.1+19.9u; alpha(u)=exp(-0.05u-4.975u²), sigma(u)=sqrt(1-alpha(u)²), score=-epsilon/sigma, and the probability-flow drift is -beta(u)(z+score)/2. Source-era encoding integrates from noise time 0.001 to 1; target-era decoding reverses that interval with a different historical condition. Noise time and historical time are distinct inputs. The source encoder's actual weights and independently trained survival state are inherited. Definitions follow [Song et al.](https://arxiv.org/abs/2011.13456).

## Intervention and geometric analysis

Attention tensors preserve layer, head and modality-token axes. Head ablation zeros a head's attention contribution; layer/modality intervention zeros that post-layer token. Logit differences use the paired target's cosine minus the strongest different-character target within the sampled gallery. Effective-modality polysemanticity uses normalized absolute changes in transported latent dimensions after input-modality zeroing; zero-effect dimensions remain undefined. High/low PI dimension interventions and their margins are exported.

Continuity and volume scripts export measured trajectories, reconstruction, composition error, sampled distance ratios, local velocity Jacobian norms, divergence and optional flow-Jacobian log determinants. These sampled measurements do not constitute proofs of global bounds or volume preservation. Semantic/group analyses require actual source annotations. PCA and t-SNE preserve occurrence IDs.

Seven-observable trajectory analysis exports all observations and explicitly identifies missing eras. Supplied palaeographic stroke counts are used when available; skeleton chain count is an identified image proxy otherwise. K-means, spectral and agglomerative clustering operate on standardized complete trajectories and export their agreement without assigning historical cluster names automatically.

Visual sensitivity uses finite patch occlusion of actual glyph images and recomputes the visual geometry block. High/low-sensitivity comparisons remove the same number of ink pixels. Supplied nonvisual features stay fixed. The optional photo-choice adapter is a three-layer CNN trained by four-choice cross entropy against frozen glyph coordinates, selected on validation questions and evaluated on held-out characters. It is separate from script-gallery CBED.

## Evaluation interpretation

Same-character pair losses and roundtrip consistency alone do not prevent representation collapse or identify a useful metric geometry. The training objective consists of the evolution and survival losses specified above. Retrieval metrics and actual learned coordinate dispersion are needed to determine whether training succeeds. Unmatched observations, missing predictions and candidates lacking verified lineages remain explicit outputs.
