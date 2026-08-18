import json
from util.paths import CONFIG_DIR, DATASETS_DIR
import numpy as np
import random
import iteround
import csv
from tqdm import tqdm
import pandas as pd
from task_generation import generate_tasks
from util.task import execute_tasks

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
        tasks = generate_tasks()
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
    output_dir = DATASETS_DIR / f"WDN/{SET_NAME}/"
    output_dir.mkdir(exist_ok=True)

    tasks = generate_tasks(2000)

    execute_tasks(tasks, cfg, output_dir_raw, repetitions=10)
    mapping = pd.read_csv(MAPFILE_PATH)
    mapping['run'] = mapping['filename'].str.extract(r"^(\d+)_\d+\.csv")[0].astype(int)
    for run, group in mapping.groupby('run'):
        dfs = []
        for filename in group['filename']:
            df = pd.read_csv(output_dir_raw / filename)
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        averaged = combined.groupby('ElapsedTime[s]', as_index=False).mean(numeric_only=True)
        averaged.to_csv(output_dir / f"run_{run:04d}.csv", index=False)