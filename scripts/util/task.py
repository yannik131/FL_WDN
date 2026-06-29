import json
from pathlib import Path 
import tempfile
import subprocess
import copy
from util.bin_selector import get_binary_path
import os 
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

EXE = get_binary_path()

def fmt(x):
    return f"{x:.3f}" if isinstance(x, float) else str(x)

class Task:
    def __init__(self, params, mapping):
        self.filename = params.pop("filename")
        self.r = params.pop("r")
        self.params = params 
        self.mapping = mapping
    
    def __apply_to_cfg(self, cfg):
        for param, keys in self.mapping.items():
            tmp_cfg = cfg
            for key in keys[:-1]:
                tmp_cfg = tmp_cfg[key]
            tmp_cfg[keys[-1]] = self.params[param]

    def run(self, cfg, output_dir):
        cfg = copy.deepcopy(cfg)
        self.__apply_to_cfg(cfg)
        out = output_dir / self.filename

        if Path(out).exists():
            return out 
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(cfg, tmp)
            tmp_path = tmp.name

        try:
            subprocess.run(
                [
                    str(EXE),
                    f"--config={tmp_path}",
                    f"--out={out}",
                    "--duration=60",
                    "--storage-interval=0.003",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return out
    
def create_mapfile(tasks, path):
    # requires all tasks to be the same
    task = tasks[0]
    with open(path, "w") as f:
        f.write("filename,r," + ",".join(task.params.keys()) + ",r\n")
        for task in tasks:
            f.write(f"{task.filename}," + ",".join(fmt(v) for v in task.params.values()) + f",{task.r}\n")
    
def _run_task(task, cfg, output_dir):
    return task.run(cfg, output_dir)
    
def execute_tasks(tasks, cfg, output_dir):
    workers = os.cpu_count()
    queue_size = workers * 2
    futures = deque()
    print(f"Number of workers: {workers}")

    with ProcessPoolExecutor(max_workers=workers) as pool, tqdm(total=len(tasks)) as pbar:
        try:
            task_iter = iter(tasks)
            for _ in range(queue_size):
                try:
                    task = next(task_iter)
                    futures.append(pool.submit(_run_task, task, cfg, output_dir))
                except StopIteration:
                    break 

            while futures:
                done = next(as_completed(futures))
                futures.remove(done)
                pbar.update(1)

                try:
                    task = next(task_iter)
                    futures.append(pool.submit(_run_task, task, cfg, output_dir))
                except StopIteration:
                    pass 
        except KeyboardInterrupt:
            print("Interrupt detected, waiting for tasks to finish...")
            pool.shutdown(wait=False, cancel_futures=True)

    
    
