import json
import subprocess
import tempfile
from pathlib import Path
from tqdm import tqdm
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from ..bin_selector import get_binary_path
import os

root_dir = Path(__file__).resolve().parent.parent
exe = get_binary_path()

with open(root_dir / "config/WDN/transformation_simple.json") as f:
    base_cfg = json.load(f)

def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)

class Task:
    def __init__(self, filename, p, f, N, r):
        self.filename = filename
        self.p = p # probability for reaction A -> B
        self.f = f # frequency of A: A = f*N, B = (1-f)*N
        self.N = N # disc count
        self.r = r # repetition (we repeat 10 times)

p_vals = [0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 1]
f_vals = [0, 0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9, 1]
N_vals = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
repetition = list(range(10))

i = 0
tasks = []
for p in p_vals:
    for f in f_vals:
        for N in N_vals:
            for r in repetition:
                tasks.append(Task(f"transform_{i:04d}.csv", p, f, N, r))
                i += 1

with open(root_dir / "datasets/WDN/simple_transformation_set.txt", "w") as file:
    file.write("Filename,p,f,N,repetition\n")
    for task in tasks:
        file.write(f"{task.filename},{fmt(task.p)},{fmt(task.f)},{task.N},{task.r}\n")

output_dir = root_dir / "datasets/WDN/simple_transformation_set/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

def run_sim(task: Task):
    cfg = json.loads(json.dumps(base_cfg)) 
    cfg["config"]["cellMembraneType"]["discCount"] = task.N
    cfg["config"]["cellMembraneType"]["discTypeDistribution"]["A"] = task.f
    cfg["config"]["cellMembraneType"]["discTypeDistribution"]["B"] = 1.0 - task.f
    cfg["config"]["reactions"][0]["probability"] = task.p
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
            cwd=root_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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
