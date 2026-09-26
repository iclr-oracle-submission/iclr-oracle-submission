"""CCAMC source-to-model adapter with original occurrence identity and labels."""

from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .ccamc import iter_occurrences
from .images import load_glyph_image
from .feature_extractor import MultimodalFeatureExtractor
from .time_encoding import ERA_TIMES

SCRIPT_ERAS = {"甲骨文": "OBI", "金文": "Bronze"}
GROUPS = {
    "賓組": "Bin",
    "宾组": "Bin",
    "出組": "Chu",
    "出组": "Chu",
    "歷組": "Li",
    "历组": "Li",
    "子組": "Zi",
    "子组": "Zi",
    "黃組": "Huang",
    "黄组": "Huang",
}
BRONZE = {
    "西周早期": "Early Western Zhou",
    "西周晚期": "Late Western Zhou",
    "春秋": "Spring&Autumn",
    "戰國": "Warring States",
    "战国": "Warring States",
}


def model_record(row):
    era = SCRIPT_ERAS.get(row["script_type"])
    subperiod = (
        GROUPS.get(row.get("version_subgroup"))
        if era == "OBI"
        else BRONZE.get(row.get("dynasty")) if era == "Bronze" else None
    )
    return dict(
        occurrence_id=row["occurrence_id"],
        char_id=row["character"],
        character=row["character"],
        era=era,
        time=ERA_TIMES[era] if era else None,
        subperiod=subperiod,
        scribal_group=row.get("version_subgroup"),
        time_resolution=(
            "era_checkpoint_default" if era else None
        ),
        source_period=row.get("period"),
        source_dynasty=row.get("dynasty"),
        image_path=row.get("image_path") or "",
        source=row.get("page_url") or row["image_url"],
        source_book=row.get("source_book"),
        input_visibility="label_free",
    )


class CCAMCSourceDataset(Dataset):
    """All source occurrences remain addressable, including unmapped/missing images.

    A model-ready item has `features` [352]; others have features=None. Explicit
    eligible_indices select the supported OBI/Bronze population for model code.
    Other original source categories are never relabeled as later model eras.
    """

    def __init__(self, data_root, records=None, raw_root=None):
        self.root = Path(data_root)
        self.raw_root = Path(raw_root) if raw_root else self.root / "raw"
        path = Path(records) if records else self.root / "metadata/occurrences.jsonl.gz"
        self.records = [model_record(r) for r in iter_occurrences(path)]
        self.extractor = MultimodalFeatureExtractor()
        self.eligible_indices = [
            i
            for i, r in enumerate(self.records)
            if r["era"]
            and r["image_path"]
            and (self.raw_root / r["image_path"]).is_file()
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = dict(self.records[index])
        path = self.raw_root / record["image_path"]
        if record["era"] and record["image_path"] and path.is_file():
            record["features"] = self.extractor.extract(
                load_glyph_image(path), time_value=record["time"]
            )
        else:
            record["features"] = None
        return record

    def statistics(self):
        return dict(
            source_occurrences=len(self),
            model_ready_occurrences=len(self.eligible_indices),
            mapped_occurrences=sum(r["era"] is not None for r in self.records),
            unsupported_category_occurrences=sum(
                r["era"] is None for r in self.records
            ),
            mapped_without_image=sum(
                r["era"] is not None and not r["image_path"] for r in self.records
            ),
        )
