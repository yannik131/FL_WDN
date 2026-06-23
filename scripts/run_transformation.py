import argparse
import json
import subprocess
import tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("p", type=float)
args = parser.parse_args()

p = args.p
if not (0.0 <= p <= 1.0):
    raise ValueError("p must be in [0, 1]")

script_dir = Path(__file__).resolve().parent
exe = script_dir.parent / "bin/cell-cli"

with open(script_dir / "../config/WDN/transformation_simple.json") as f:
    cfg = json.load(f)

cfg["config"]["reactions"][0]["probability"] = p

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
    json.dump(cfg, tmp)
    tmp_path = tmp.name

subprocess.run(
    [
        exe,
        f"--config={tmp_path}",
        f"--out=../datasets/WDN/transformation_simple_{p}.csv",
        "--duration=60",
        "--storage-interval=0.003",
    ],
    cwd=script_dir,
    check=True,
)
