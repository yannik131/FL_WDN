import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque
from tqdm import tqdm
import subprocess
from util.paths import get_binary_path, CONFIG_DIR, DATASETS_DIR, RESULTS_DIR
import pandas as pd

OUTPUT_DIR = DATASETS_DIR / "WDN/water_averaged/"

def run_task(filename):
    subprocess.run(
        [
            str(get_binary_path()),
            f"--config={CONFIG_DIR / 'WDN/water.json'}",
            f"--out={OUTPUT_DIR / filename}",
            "--duration=60",
            "--storage-interval=0.003",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

def run_tasks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    workers = os.cpu_count()
    futures = deque()
    print(f"Number of workers: {workers}")
    N = 1000
    if len(list(OUTPUT_DIR.glob('*.csv'))) == 1000:
        return

    with ProcessPoolExecutor(max_workers=workers) as pool, tqdm(total=N) as pbar:
        try:
            for i in range(N):
                filename = f"run_{i:04d}.csv"
                try:
                    futures.append(pool.submit(run_task, filename))
                except StopIteration:
                    break

            while futures:
                done = next(as_completed(futures))
                futures.remove(done)
                pbar.update(1)
        except KeyboardInterrupt:
            print("Interrupt detected, waiting for tasks to finish...")
            pool.shutdown(wait=False, cancel_futures=True)

def save_average_trajectory():
    run_tasks()
    dfs = [pd.read_csv(file) for file in OUTPUT_DIR.glob("*.csv")]
    df = pd.concat(dfs, ignore_index=True)
    average = df.groupby("ElapsedTime[s]").mean(numeric_only=True)
    average.to_csv(RESULTS_DIR / "WDN/water_averaged_df.csv")

if __name__ == '__main__':
    save_average_trajectory()