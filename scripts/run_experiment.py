#!/usr/bin/env python3
"""Resolve and run a registered experiment; missing inputs fail before execution."""
import argparse, json, sys, subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id")
    p.add_argument("--list", action="store_true")
    p.add_argument("--inputs", help="JSON object of required path variables")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--execute", action="store_true")
    p.add_argument("extra", nargs=argparse.REMAINDER)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "experiments.json").read_text())
    if a.list:
        for row in registry["experiments"]:
            print(row["id"] + "\t" + row["paper_section"])
        return
    row = next((r for r in registry["experiments"] if r["id"] == a.id), None)
    if row is None:
        p.error("Unknown experiment ID")
    values = json.loads(Path(a.inputs).read_text()) if a.inputs else {}
    values["seed"] = a.seed
    missing = [k for k in row["required_inputs"] if k not in values]
    if missing:
        p.error("Missing required input variables: " + ", ".join(missing))
    for key in row["required_inputs"]:
        path = Path(values[key]).expanduser().resolve()
        if not path.exists():
            p.error("Input path absent: " + key)
        values[key] = str(path)
    if "output" in values:
        values["output"] = str(Path(values["output"]).expanduser().resolve())
    if not row["arguments"]:
        if not a.extra:
            p.error(
                "This analysis requires explicit CLI arguments after --; see "
                + row["entrypoint"]
                + " --help"
            )
        args = []
    else:
        try:
            args = [v.format_map(values) for v in row["arguments"]]
        except KeyError as e:
            p.error("Missing command variable " + str(e))
    extra = a.extra[1:] if a.extra[:1] == ["--"] else a.extra
    cmd = [sys.executable, str(root / row["entrypoint"]), *args, *extra]
    print(
        json.dumps(
            dict(experiment=a.id, command=cmd, protocol=row["protocol"]),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if a.execute:
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
