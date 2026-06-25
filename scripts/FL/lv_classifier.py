from scipy.signal import savgol_filter, find_peaks
import pandas as pd
import numpy as np

def has_lv_dynamics(df: pd.DataFrame):
    W = 500
    t, prey, predator = df['ElapsedTime[s]'], savgol_filter(df['Prey'], W, 3), savgol_filter(df['Predator'], W, 3)

    prey_peaks, _ = find_peaks(prey, prominence=50)
    predator_peaks, _ = find_peaks(predator, prominence=50)

    N_prey = len(prey_peaks)
    N_pred = len(predator_peaks)

    if N_prey < 3 or N_pred < 3 or abs(N_prey - N_pred) > 1:
        return False 

    prey_t = t.iloc[prey_peaks].to_numpy()
    pred_t = t.iloc[predator_peaks].to_numpy()

    lags = []
    for pt in prey_t:
        idx = np.argmin(np.abs(pred_t - pt))
        lags.append(pred_t[idx] - pt)

    lags = np.array(lags)
    if np.sum(lags < 0) > 1:
        return False

    return True 
