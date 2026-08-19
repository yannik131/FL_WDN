import json
from util.paths import CONFIG_DIR, DATASETS_DIR
import numpy as np
import random
import iteround
import csv
from tqdm import tqdm
import pandas as pd
from WDN.task_generation import generate_tasks
from util.task import execute_tasks
import re

random.seed(42)
np.random.seed(42)

SET_NAME = 'full_set_4'
MAPFILE_PATH = DATASETS_DIR / f"WDN/{SET_NAME}_raw.csv"
IMAGE_DIR = DATASETS_DIR / f"WDN/{SET_NAME}_figs/"

def create_mapfile(plot=False):
    IMAGE_DIR.mkdir(exist_ok=True)
    tasks = []
    with open(MAPFILE_PATH, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "species", "masses", "fractions", "reactions"])
        tasks = generate_tasks(2000)
        for task in tqdm(tasks):
            if plot:
                task.plot(IMAGE_DIR)
            rounded_fractions = iteround.saferound(task.fractions, 3)
            reactions = []
            for reaction in task.reactions:
                reactions.append([reaction['indices'], reaction['p']])

            writer.writerow([
                task.filename,
                json.dumps(task.species),
                json.dumps(task.masses),
                json.dumps(rounded_fractions),
                json.dumps(reactions),
            ])

def run_tasks():
    with open(CONFIG_DIR / "WDN/trans_comp.json") as f:
        cfg = json.load(f)

    output_dir_raw = DATASETS_DIR / f"WDN/{SET_NAME}_raw"
    output_dir_raw.mkdir(exist_ok=True)

    tasks = generate_tasks(2000)
    execute_tasks(tasks, cfg, output_dir_raw, repetitions=10)

def average_outputs():
    output_dir = DATASETS_DIR / f"WDN/{SET_NAME}/"
    output_dir.mkdir(exist_ok=True, parents=True)
    csvs = list(output_dir.glob('*.csv'))
    groups = {}
    for path in csvs:
        match = re.match(r'(.+)_(\d+)\.csv$', path.name)
        if not match:
            raise RuntimeError(f"No match for file {path.name}")
        base_name = match.group(1)
        groups.setdefault(base_name, []).append(path)
    for base_name, paths in groups.items():
        dfs = [pd.read_csv(path) for path in paths]
        if len(dfs) > 1:
            combined = pd.concat(dfs, ignore_index=True)
            averaged = combined.groupby('ElapsedTime[s]', as_index=False).mean(numeric_only=True)
        else:
            averaged = dfs[0]
        averaged.to_csv(output_dir / f'{base_name}.csv', index=False)

if __name__ == '__main__':
    # run_tasks()
    # average_outputs()
    create_mapfile(plot=True)
    