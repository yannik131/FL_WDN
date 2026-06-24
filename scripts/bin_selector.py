import platform
from pathlib import Path

def get_binary_path():
    script_dir = Path(__file__).resolve().parent
    bin_dir = script_dir.parent / "bin"

    system = platform.system().lower()

    if system == "linux":
        binary = "cell-cli-linux"
    elif system == "darwin":
        binary = "cell-cli-mac"
    elif system == "windows":
        binary = "cell-cli.exe"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

    return str(bin_dir / binary)