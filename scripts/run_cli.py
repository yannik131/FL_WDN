import subprocess
from pathlib import Path

script_dir = Path(__file__).resolve().parent
subprocess.run(
    [
        "../bin/cell-cli.exe",
        "--config=../config/preyPredator.json",
        "--out=../datasets/example/bad.csv",
        "--duration=60",
        "--storage-interval=0.003",
    ],
    cwd=script_dir,
    check=True,
)