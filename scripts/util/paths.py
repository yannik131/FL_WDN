from pathlib import Path 
import platform

ROOT_DIR = Path(__file__).resolve().parent.parent.parent 
CONFIG_DIR = ROOT_DIR / "config/"
DATASETS_DIR = ROOT_DIR / "datasets/"
MODELS_DIR = ROOT_DIR / "models/"
RESULTS_DIR = ROOT_DIR / "results/"

def get_binary_path():
    system = platform.system().lower()

    if system == "linux":
        binary = "cell-cli-linux"
    elif system == "darwin":
        binary = "cell-cli-mac"
    elif system == "windows":
        binary = "cell-cli.exe"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    return ROOT_DIR / "bin" / binary
