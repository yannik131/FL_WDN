import subprocess
from pathlib import Path

script_dir = Path(__file__).resolve().parent
exe_dir = script_dir.parent / "bin/cell-cli.exe"
subprocess.run(
    [
        exe_dir,
        "--config=../config/preyPredator.json",
        "--out=../datasets/example/medium.csv",
        "--duration=60",
        "--storage-interval=0.003",
    ],
    cwd=script_dir,
    check=True,
)