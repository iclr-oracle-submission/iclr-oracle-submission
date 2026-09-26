"""Load raster or SVG glyphs on an opaque white canvas."""

from io import BytesIO
from pathlib import Path
import os
import sys
import numpy as np
from PIL import Image


def load_glyph_image(path, size=128):
    path = Path(path)
    if path.suffix.lower() == ".svg":
        if sys.platform == "darwin":
            # Homebrew Cairo can live outside the system dynamic-library search path.
            search = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":")
            for directory in ("/opt/homebrew/lib", "/usr/local/lib"):
                if (
                    Path(directory) / "libcairo.2.dylib"
                ).is_file() and directory not in search:
                    search.append(directory)
            os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(p for p in search if p)
        import cairosvg

        stream = BytesIO(
            cairosvg.svg2png(
                bytestring=path.read_bytes(),
                url=path.resolve().as_uri(),
                output_width=size,
                output_height=size,
            )
        )
    else:
        stream = path
    with Image.open(stream) as source:
        rgba = source.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return np.asarray(Image.alpha_composite(background, rgba).convert("L"))
