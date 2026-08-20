import numpy as np
import pandas as pd

def resample_counts(df: pd.DataFrame, dt=0.05, t_max=None):
    time_col = "ElapsedTime[s]"
    x = df[time_col].to_numpy()
    if t_max is None:
        t_max = x.max()
    new_time = np.arange(x.min(), t_max + dt, dt)
    count_cols = df.columns.drop(time_col)

    df_interp = pd.DataFrame({time_col: new_time})
    for col in count_cols:
        df_interp[col] = np.interp(new_time, x, df[col].to_numpy())

    N = df_interp.loc[0, count_cols].sum()
    if N > 0:
        df_interp[count_cols] /= N

    return df_interp