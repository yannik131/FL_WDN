import pandas as pd
from .lv_classifier import has_lv_dynamics
from tqdm import tqdm
from util.paths import DATASETS_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed

csv_dir = DATASETS_DIR / "FL/full_lv_set/"
mapping = pd.read_csv(DATASETS_DIR / "FL/full_lv_set.csv")

rows_iter = list(mapping.itertuples(index=True))
rows = []

def process_row(row):
    df = pd.read_csv(DATASETS_DIR / f"FL/full_lv_set/{row.filename}")
    is_lv = has_lv_dynamics(df)
    return (row.p1, row.p2, row.p3, row.p4, row.p5, row.p6, int(is_lv))

with ProcessPoolExecutor(max_workers=4) as pool:
    try:
        futures = [pool.submit(process_row, r) for r in rows_iter]

        for f in tqdm(as_completed(futures), total=len(futures)):
            rows.append(f.result())

        df_all = pd.DataFrame(rows, columns=["p1", "p2", "p3", "p4", "p5", "p6", "is_lv"])
        df_all.to_csv(DATASETS_DIR / "FL/lv_heat_map_full_df.csv", index=False)
    except KeyboardInterrupt:
        print("Shutting down pool..")
        pool.shutdown(wait=True, cancel_futures=True)