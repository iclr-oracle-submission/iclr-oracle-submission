"""Offline five-modality features (Appendix L).

No character ID is used as a feature. Learned 42/64/16D projections and CNN
blocks are supplied as frozen, provenance-recorded vectors. The visual block
uses the deterministic contour, skeleton, topology and density descriptors
specified in IMPLEMENTATION.md.
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize


def vector(value, dim, name):
    if value is None:
        return np.zeros(dim, dtype=np.float32)
    a = np.asarray(value, dtype=np.float32)
    if a.shape != (dim,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite {dim}D vector")
    return a


def positional(value, dim=16):
    freq = np.exp(-np.log(10000.0) * np.arange(0, dim, 2) / dim)
    out = np.empty(dim, dtype=np.float32)
    out[0::2], out[1::2] = np.sin(value * freq), np.cos(value * freq)
    return out


class VisualFeatureExtractor:
    feature_dim = 128

    def extract(self, image, cnn_density=None):
        image = np.asarray(image)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.ndim != 2 or not image.size:
            raise ValueError("Require glyph raster")
        image = cv2.resize(
            image.astype(np.uint8), (64, 64), interpolation=cv2.INTER_AREA
        )
        _, ink = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = ink.astype(np.uint8)
        if ink.mean() > 0.5:
            ink = 1 - ink
        contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        fourier = np.zeros(16)
        curve = np.zeros(16)
        if contours:
            c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
            if len(c) >= 3:
                c -= c.mean(0)
                f = np.abs(np.fft.fft(c[:, 0] + 1j * c[:, 1]))[1:17]
                fourier[: len(f)] = f / max(np.linalg.norm(f), 1e-8)
                v = np.roll(c, -1, axis=0) - c
                angles = np.arctan2(v[:, 1], v[:, 0])
                delta = (np.roll(angles, -1) - angles + np.pi) % (2 * np.pi) - np.pi
                curve = np.histogram(delta, bins=16, range=(-np.pi, np.pi))[0] / len(
                    delta
                )
        skel = skeletonize(ink.astype(bool)).astype(np.uint8)
        gx, gy = cv2.Sobel(ink.astype(float), cv2.CV_64F, 1, 0), cv2.Sobel(
            ink.astype(float), cv2.CV_64F, 0, 1
        )
        angle = np.arctan2(gy, gx)
        hist = np.histogram(
            angle, bins=16, range=(-np.pi, np.pi), weights=np.hypot(gx, gy)
        )[0]
        hist /= max(hist.sum(), 1e-8)
        stroke_grid = np.array(
            [
                skel[y : y + 16, x : x + 16].mean()
                for y in range(0, 64, 16)
                for x in range(0, 64, 16)
            ]
        )
        nlabels, _, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        _, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = 0 if hier is None else int((hier[0, :, 3] >= 0).sum())
        neighbors = cv2.filter2D(skel, cv2.CV_16S, np.ones((3, 3), np.int16)) - skel
        top = np.array(
            [
                nlabels - 1,
                holes,
                nlabels - 1 - holes,
                ((neighbors == 1) & (skel == 1)).sum(),
                ((neighbors >= 3) & (skel == 1)).sum(),
                skel.sum() / 4096,
                ink.mean(),
                *(
                    np.quantile(
                        stats[1:, cv2.CC_STAT_AREA] / 4096, [0, 0.25, 0.5, 0.75, 1]
                    )
                    if nlabels > 1
                    else [0] * 5
                ),
                0,
                0,
                0,
                0,
            ],
            dtype=float,
        )
        # Topology: 12 measurements + 4 normalized bbox coordinates.
        ys, xs = np.where(ink)
        if len(xs):
            top[-4:] = [xs.min() / 64, ys.min() / 64, xs.max() / 64, ys.max() / 64]
        symmetry = []
        for transform in (
            np.fliplr,
            np.flipud,
            lambda a: np.rot90(a, 2),
            lambda a: np.rot90(a),
        ):
            b = transform(ink)
            for y in range(0, 64, 32):
                for x in range(0, 64, 32):
                    symmetry.append(
                        1
                        - np.abs(
                            ink[y : y + 32, x : x + 32].astype(float)
                            - b[y : y + 32, x : x + 32]
                        ).mean()
                    )
        density = np.array(
            [
                ink[y : y + 16, x : x + 16].mean()
                for y in range(0, 64, 16)
                for x in range(0, 64, 16)
            ]
        )
        return vector(
            np.concatenate(
                [
                    fourier,
                    curve,
                    hist,
                    stroke_grid,
                    top,
                    symmetry,
                    density,
                    vector(cnn_density, 16, "cnn_density"),
                ]
            ),
            128,
            "visual",
        )


class StructuralFeatureExtractor:
    feature_dim = 64

    def extract(self, formation_type=None, component_embedding=None, layout=None):
        formation = np.zeros(6, dtype=np.float32)
        if formation_type is not None:
            if not isinstance(formation_type, int) or not 0 <= formation_type < 6:
                raise ValueError(
                    "formation_type must be an annotated liushu index in [0,5]"
                )
            formation[formation_type] = 1
        return np.concatenate(
            [
                formation,
                vector(component_embedding, 42, "component_embedding"),
                vector(layout, 16, "layout"),
            ]
        )


class SemanticFeatureExtractor:
    feature_dim = 64

    def extract(self, embedding=None):
        return vector(embedding, 64, "BERT definition projection")


class ContextFeatureExtractor:
    feature_dim = 64

    def extract(self, cooccurring_features=None, projection=None):
        if cooccurring_features is None or len(cooccurring_features) == 0:
            return np.zeros(64, dtype=np.float32)
        # Appendix L context descriptor: project concatenated visual/structural/semantic
        # vectors of OTHER occurrences from the SAME artifact; then average.
        a = np.stack(
            [vector(v, 256, "cooccurring feature") for v in cooccurring_features]
        )
        p = np.asarray(projection, dtype=np.float32)
        if p.shape != (256, 64) or not np.isfinite(p).all():
            raise ValueError(
                "Require frozen 256x64 context projection fitted on training data"
            )
        return (a.mean(0) @ p).astype(np.float32)


class SpatiotemporalFeatureExtractor:
    feature_dim = 32

    def extract(self, time_value=None, coordinates=None):
        if time_value is not None and (
            not np.isfinite(time_value) or not 0 <= time_value <= 1
        ):
            raise ValueError("Time value must be finite and in [0,1]")
        temporal = (
            np.zeros(16, dtype=np.float32)
            if time_value is None
            else positional(float(time_value) * 10000)
        )
        spatial = np.zeros(16, dtype=np.float32)
        if coordinates is not None:
            lat, lon = coordinates
            if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError("Invalid lat/lon")
            spatial = np.concatenate(
                [positional(np.deg2rad(lat), 8), positional(np.deg2rad(lon), 8)]
            )
        return np.concatenate([temporal, spatial])


class MultimodalFeatureExtractor:
    feature_dim = 352

    def __init__(self):
        self.visual_extractor = VisualFeatureExtractor()
        self.structural_extractor = StructuralFeatureExtractor()
        self.semantic_extractor = SemanticFeatureExtractor()
        self.context_extractor = ContextFeatureExtractor()
        self.spatiotemporal_extractor = SpatiotemporalFeatureExtractor()

    def extract(
        self,
        image,
        time_value=None,
        *,
        formation_type=None,
        component_embedding=None,
        layout=None,
        semantic_embedding=None,
        cooccurring_features=None,
        context_projection=None,
        coordinates=None,
        cnn_density=None,
    ):
        return vector(
            np.concatenate(
                [
                    self.visual_extractor.extract(image, cnn_density),
                    self.structural_extractor.extract(
                        formation_type, component_embedding, layout
                    ),
                    self.semantic_extractor.extract(semantic_embedding),
                    self.context_extractor.extract(
                        cooccurring_features, context_projection
                    ),
                    self.spatiotemporal_extractor.extract(time_value, coordinates),
                ]
            ),
            352,
            "multimodal",
        )

    def get_feature_indices(self):
        return {
            "visual": (0, 128),
            "structural": (128, 192),
            "semantic": (192, 256),
            "context": (256, 320),
            "spatiotemporal": (320, 352),
        }
