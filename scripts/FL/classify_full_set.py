import pandas as pd
from FL.lv_classifier import has_lv_dynamics
from tqdm import tqdm
from util.paths import DATASETS_DIR
from concurrent.futures import ProcessPoolExecutor, as_completed

set_name = "full_lv_set"
csv_dir = DATASETS_DIR / f"FL/{set_name}/"
mapping = pd.read_csv(DATASETS_DIR / f"FL/{set_name}.csv")

rows_iter = list(mapping.itertuples(index=False, name=None))
rows = []

def process_row(row):
    filename, p1, p2, p3, p4, p5, p6, r = row
    df = pd.read_csv(DATASETS_DIR / f"FL/{set_name}/{filename}")
    is_lv = has_lv_dynamics(df)
    return (p1, p2, p3, p4, p5, p6, int(is_lv))

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        try:
            futures = [pool.submit(process_row, r) for r in rows_iter]

            for f in tqdm(as_completed(futures), total=len(futures)):
                rows.append(f.result())

            df_all = pd.DataFrame(rows, columns=["p1", "p2", "p3", "p4", "p5", "p6", "is_lv"])
            df_all.to_csv(DATASETS_DIR / "FL/lv_heat_map_full.csv", index=False)
        except KeyboardInterrupt:
            print("Shutting down pool..")
            pool.shutdown(wait=True, cancel_futures=True)
