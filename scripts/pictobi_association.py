#!/usr/bin/env python3
"""Train/evaluate a conventional photo adapter for source-bearing four-choice tasks."""
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import load_model
from models.photo_association import PhotoAssociation
from data.images import load_glyph_image
from data.feature_extractor import MultimodalFeatureExtractor


def load_rows(path):
    rows = json.loads(Path(path).read_text())
    ids = [r["query_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question identity")
    groups = {}
    for r in rows:
        if (
            r["split"] not in ("train", "val", "test")
            or len(r["options"]) != 4
            or len(set(o["key"] for o in r["options"])) != 4
        ):
            raise ValueError("Invalid question or split")
        if r["correct_answer"] not in [o["key"] for o in r["options"]] or not r.get(
            "source"
        ):
            raise ValueError("Missing answer or source")
        if groups.setdefault(r["char_id"], r["split"]) != r["split"]:
            raise ValueError("Character leakage across question splits")
        if r.get("input_visibility") != "label_free":
            raise ValueError("Query observations must be label_free")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "evaluate"])
    for k in ("encoder_checkpoint", "manifest", "output_dir"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--adapter_checkpoint")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.epochs < 1 or a.batch_size < 1:
        p.error("Positive epochs and batch size required")
    root = Path(a.manifest).resolve().parent
    rows = load_rows(a.manifest)
    encoder, cfg = load_model(a.encoder_checkpoint, a.device)
    adapter = PhotoAssociation(cfg.manifold_dim).to(a.device)
    extract = MultimodalFeatureExtractor()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def batch(indices):
        selected = [rows[i] for i in indices]
        features = []
        photos = []
        for r in selected:
            f = (
                np.load(root / r["features"])
                if r.get("features")
                else extract.extract(
                    load_glyph_image(root / r["query_image"]),
                    time_value=float(r.get("time", 0.05)),
                )
            )
            if f.shape != (352,) or not np.isfinite(f).all():
                raise ValueError("Invalid query features")
            features.append(f)
            options = []
            for o in r["options"]:
                with Image.open(root / o["image"]) as image:
                    options.append(
                        np.asarray(
                            image.convert("RGB").resize((128, 128)), dtype=np.float32
                        ).transpose(2, 0, 1)
                        / 255.0
                    )
            photos.append(options)
        x = torch.tensor(np.array(features), device=a.device)
        t = x.new_tensor([r.get("time", 0.05) for r in selected])
        with torch.no_grad():
            glyph = encoder.encode(x, t)
        logits = adapter(glyph, torch.tensor(np.array(photos), device=a.device))
        target = torch.tensor(
            [
                [o["key"] for o in r["options"]].index(r["correct_answer"])
                for r in selected
            ],
            device=a.device,
        )
        return logits, target, selected

    if a.mode == "train":
        train = [i for i, r in enumerate(rows) if r["split"] == "train"]
        val = [i for i, r in enumerate(rows) if r["split"] == "val"]
        if not train or not val:
            raise ValueError("Training and validation questions required")
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=a.lr, weight_decay=1e-4)
        best = float("inf")
        for epoch in range(1, a.epochs + 1):
            adapter.train()
            total = 0
            order = np.random.permutation(train)
            for start in range(0, len(order), a.batch_size):
                ids = order[start : start + a.batch_size].tolist()
                optimizer.zero_grad()
                logits, labels, _ = batch(ids)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(ids)
            adapter.eval()
            validation = 0
            with torch.no_grad():
                for start in range(0, len(val), a.batch_size):
                    ids = val[start : start + a.batch_size]
                    logits, labels, _ = batch(ids)
                    validation += float(
                        torch.nn.functional.cross_entropy(logits, labels)
                    ) * len(ids)
            validation /= len(val)
            if validation < best:
                best = validation
                torch.save(
                    dict(
                        model_state_dict=adapter.state_dict(),
                        dimension=cfg.manifold_dim,
                        encoder_state=encoder.state_dict(),
                        encoder_config=vars(cfg),
                        encoder_checkpoint=str(Path(a.encoder_checkpoint).resolve()),
                        training_characters=sorted(
                            {rows[i]["char_id"] for i in train + val}
                        ),
                        seed=a.seed,
                        epoch=epoch,
                        validation_loss=validation,
                        training_ids=[rows[i]["query_id"] for i in train],
                        validation_ids=[rows[i]["query_id"] for i in val],
                    ),
                    out / "best_adapter.pt",
                )
            with (out / "history.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            epoch=epoch,
                            train_loss=total / len(train),
                            val_loss=validation,
                        )
                    )
                    + "\n"
                )
    else:
        if not a.adapter_checkpoint:
            p.error("--adapter_checkpoint required")
        state = torch.load(
            a.adapter_checkpoint, map_location=a.device, weights_only=False
        )
        if (
            state["dimension"] != cfg.manifold_dim
            or state["encoder_config"] != vars(cfg)
            or any(
                not torch.equal(v, state["encoder_state"][k].to(v))
                for k, v in encoder.state_dict().items()
            )
        ):
            raise ValueError("Adapter requires its exact frozen encoder")
        test = [i for i, r in enumerate(rows) if r["split"] == "test"]
        if not test:
            raise ValueError("No test questions")
        forbidden = set(state["training_ids"] + state["validation_ids"])
        if any(
            rows[i]["query_id"] in forbidden
            or rows[i]["char_id"] in state["training_characters"]
            for i in test
        ):
            raise ValueError("Evaluation question overlaps adapter supervision")
        adapter.load_state_dict(state["model_state_dict"])
        adapter.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(test), a.batch_size):
                logits, labels, selected = batch(test[start : start + a.batch_size])
                probs = logits.softmax(-1)
                for j, r in enumerate(selected):
                    winner = int(logits[j].argmax())
                    predictions.append(
                        dict(
                            query_id=r["query_id"],
                            question_index=r["question_index"],
                            choice=r["options"][winner]["key"],
                            confidence=float(probs[j, winner]),
                            choice_scores={
                                o["key"]: float(logits[j, k])
                                for k, o in enumerate(r["options"])
                            },
                        )
                    )
        (out / "predictions.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
        )


if __name__ == "__main__":
    main()
