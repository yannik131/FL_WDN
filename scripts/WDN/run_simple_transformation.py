import json
import subprocess
import tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # project root
from bin_selector import get_binary_path

root_dir = Path(__file__).resolve().parent.parent.parent
result_dir = root_dir / "results/WDN/simple_transformation/ex2/"
exe = get_binary_path()

with open(root_dir / "config/WDN/transformation_simple.json") as f:
    base_cfg = json.load(f)


def run_once(A0, B0, p):
    N = A0 + B0
    cfg = json.loads(json.dumps(base_cfg))
    cfg["config"]["cellMembraneType"]["discCount"] = N
    cfg["config"]["cellMembraneType"]["discTypeDistribution"]["A"] = A0 / N
    cfg["config"]["cellMembraneType"]["discTypeDistribution"]["B"] = B0 / N
    cfg["config"]["reactions"][0]["probability"] = p

    cfg_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(cfg, cfg_file)
    cfg_file.close()
    cfg_path = cfg_file.name

    out_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    out_file.close()
    out_path = Path(out_file.name)

    try:
        subprocess.run(
            [
                str(exe),
                f"--config={cfg_path}",
                f"--out={out_path}",
                "--duration=60",
                "--storage-interval=0.003",
            ],
            cwd=root_dir,
            check=True,
        )

        df = pd.read_csv(out_path)
        df.to_csv(result_dir / f"sim_{A0}_{B0}_{p}.csv", index=False)
        df.plot(x="ElapsedTime[s]")
        plt.title(f"Simulated: A0={A0}, B0={B0}, p={p}")
        plt.show()

    finally:
        Path(cfg_path).unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run_once(A0=1000, B0=0, p=0.05)