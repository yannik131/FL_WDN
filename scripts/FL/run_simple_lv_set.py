import json
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from bin_selector import get_binary_path
from collections import deque

script_dir = Path(__file__).resolve().parent
exe = get_binary_path()

with open(script_dir / "../config/FL/preyPredator.json") as f:
    base_cfg = json.load(f)

def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)

class Task:
    def __init__(self, filename, p1, p2, r):
        self.filename = filename
        self.p1 = p1 
        self.p2 = p2 
        self.r = r

p1_vals = np.linspace(0, 1, 101)
p2_vals = np.linspace(0, 1, 101)
repetition = list(range(4))

i = 0
tasks = []
for p1 in p1_vals:
    for p2 in p2_vals:
        for r in repetition:
            tasks.append(Task(f"lv_simple_{i:04d}.csv", p1, p2, r))
            i += 1

with open(script_dir / "../datasets/FL/simple_lv_set.txt", "w") as file:
    file.write("Filename,p1,p2,repetition\n")
    for task in tasks:
        file.write(f"{task.filename},{fmt(task.p1)},{fmt(task.p2)},{task.r}\n")

output_dir = script_dir / "../datasets/FL/simple_lv_set/"
Path(output_dir).mkdir(parents=True, exist_ok=True)


def run_sim(task: Task):
    cfg = json.loads(json.dumps(base_cfg))

    cfg["config"]["reactions"][0]["probability"] = task.p1
    cfg["config"]["reactions"][1]["probability"] = task.p2
    out = output_dir / task.filename

    if Path(out).exists():
        return out

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
        tmp_path = tmp.name

    try:
        subprocess.run(
            [
                str(exe),
                f"--config={tmp_path}",
                f"--out={out}",
                "--duration=60",
                "--storage-interval=0.003",
            ],
            cwd=script_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return out


if __name__ == "__main__":
    workers = os.cpu_count()
    queue_size = workers * 2
    futures = deque()
    print(f"Number of workers: {workers}")

    with ProcessPoolExecutor(max_workers=workers) as pool, tqdm(total=len(tasks)) as pbar:
        try:
            task_iter = iter(tasks)
            for _ in range(queue_size):
                try:
                    futures.append(pool.submit(run_sim, next(task_iter)))
                except StopIteration:
                    break 

            while futures:
                done = next(as_completed(futures))
                futures.remove(done)
                pbar.update(1)

                try:
                    futures.append(pool.submit(run_sim, next(task_iter)))
                except StopIteration:
                    pass 
        except KeyboardInterrupt:
            print("Interrupt detected, waiting for tasks to finish...")
            pool.shutdown(wait=False, cancel_futures=True)
