import numpy as np
from .functions import isnt_number
import pandas as pd
from sklearn.cluster import KMeans
from functools import partial
from tqdm import tqdm
from collections import defaultdict
# from tsmoothie import ConvolutionSmoother


def qa_numeric(variable):
    isnt_numeric = np.vectorize(isnt_number, otypes=[bool])
    return isnt_numeric(variable)


def qa_bounds(variable, bounds):
    v = pd.to_numeric(variable, errors='coerce')
    v[qa_numeric(v)] = np.nan
    out = np.zeros(len(v), dtype=bool)
    out[~np.isnan(v)] = np.logical_or(v[~np.isnan(v)] < float(bounds[0]), v[~np.isnan(v)] > float(bounds[1]))
    return out


def qa_edges(variable, time, edges):
    out = np.zeros(len(variable), dtype=bool)
    out[time > time[-1] - edges] = True
    out[time < time[0] + edges] = True
    return out


def qa_iqr(arr, factor=3):
    """
    Remove values outside interquartile Range (IQR) for timeseries data.

    Parameters:
        arr (np.array): Data array to which to apply the quality assurance
        factor (int): threshold for outlier labelling rule

    Returns:
        flag (np.array): An array of bools where True means non-trusted data after this outlier detection
    """
    y = np.copy(arr)
    flag = np.ones(len(y), dtype=bool)
    q75 = np.quantile(y, 0.75)
    q25 = np.quantile(y, 0.25)
    iqr = q75 - q25
    if iqr != 0:
        sup = q75 + factor * iqr
        inf = q25 - factor * iqr
        idx1 = np.where(y >= sup)[0]
        idx2 = np.where(y <= inf)[0]
        idx0 = np.r_[idx1, idx2]
    else:
        idx0 = np.array([])

    flag[idx0] = True
    return flag


def qa_variation_rate(arr):
    """
    Remove values if variation rate exceed a defined threshold.

    Parameters:
        arr (np.array): Data array to which to apply the quality assurance

    Returns:
        flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
    """
    y = np.copy(arr)
    flag = np.ones(len(y), dtype=bool)
    vec_diff = abs(np.diff(y))
    if len(vec_diff) > 0:
        vecdiff_quan = np.quantile(vec_diff, 0.999)
        idx_vecdiff = np.where(vec_diff >= vecdiff_quan)[0]

        vec_max = np.max(y)
        if vec_max / np.quantile(y, 0.99) < 2:
            quantile_threshold = 0.99
        elif vec_max / np.quantile(y, 0.999) < 2:
            quantile_threshold = 0.999
        elif vec_max / np.quantile(y, 0.9999) < 2:
            quantile_threshold = 0.9999
        elif vec_max / np.quantile(y, 0.99999) < 2:
            quantile_threshold = 0.99999

        vec_quan = np.quantile(y, quantile_threshold)
        idx_vec = np.where(y >= vec_quan)[0]
        idx = list(set(idx_vecdiff) & set(idx_vec))
    else:
        idx = []

    flag[idx] = True
    return flag


def qa_iqr_moving(arr, window_size=15, factor=3):
    """
   Remove outliers values based on Interquartile Range (IQR) for a window of time series data

   Parameters:
       ARR (np.array): Data array to which to apply the quality assurance
       windowsize (np.int): window size of data

   Returns:
       flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
   """
    y = np.copy(arr)
    flag = np.ones(len(y), dtype=bool)

    if len(y) < window_size:
        print("ERROR! Window size is larger than array length.")
    else:
        for i in range(0, (len(y) - window_size + 1)):
            y_sub = np.copy(y[i:i + window_size])
            q75 = np.quantile(y_sub, 0.75)
            q25 = np.quantile(y_sub, 0.25)
            IQR = q75 - q25
            outsup = q75 + factor * IQR
            outinf = q25 - factor * IQR

            idx1 = np.where(y_sub >= outsup)[0]
            idx2 = np.where(y_sub <= outinf)[0]

            idx0 = np.r_[idx1, idx2]

            if len(idx0) != 0:
                y_qual[i + idx0] = y_qual[i + idx0] + 1

    y_qual = np.array(y_qual, dtype=bool)

    return y_qual


def qa_max(vecX, vecX_qual, factor1=3, semiwindow=1000):
    """
       Remove outliers values based on Interquartile Range (IQR) for a window of time series data

       Parameters:
           vec (np.array): Data array to which to apply the quality assurance
           vecX_qual (np.array): An array of bools where True means non-trusted data
           semiwindow (int): window size of data
           factor1 (int): threshold for outlier labelling rule

       Returns:
           y_qual (np.array): An array of bools where True means non-trusted data for this outlier dectection
   """

    y = np.copy(vecX)
    y_qual = np.copy(vecX_qual)

    maxy = np.nanmax(y)

    n0 = np.where(y == maxy)[0][0] - semiwindow
    n1 = np.where(y == maxy)[0][0] + semiwindow

    if n0 < 0:
        n0 = 0
    if n1 > (len(y) - 1):
        n1 = len(y) - 1

    vec_sub = y[n0:n1]
    vec_qual = np.zeros(len(vec_sub))

    y_qual99 = qa_iqr(vec_sub, vec_qual, factor=factor1)
    if sum(y_qual99) > 0:
        y_qual[n0:n1] = y_qual[n0:n1] + y_qual99  # update vec_qual

    y_qual = np.array(y_qual, dtype=bool)
    return y_qual


def qa_convolution(vecX, vecX_qual, window_len=30, window_type='blackman', n_sigma=2, threshold=20):
    """
        Remove values using convolutional smoothing of single or multiple time-series

        Parameters:
            vecX (np.array): Data array to which to apply the quality assurance
            vecX_qual (np.array): An array of bools where True means non-trusted data
            window_len (int) : Greater than equal to 1. The length of the window used to compute
    the convolutions.
            window_type (str):  The type of the window used to compute the convolutions.
    Supported types are: 'ones', 'hanning', 'hamming', 'bartlett', 'blackman'.

        Returns:
            y_qual (np.array): An array of bools where True means non-trusted data for this outlier detection
    """

    timesteps = len(vecX)

    series = defaultdict(partial(np.ndarray, shape=(1), dtype='float32'))

    for i in tqdm(range(timesteps + 1), total=(timesteps + 1)):
        if i > window_len:
            smoother = ConvolutionSmoother(window_len=window_len, window_type=window_type)
            smoother.smooth(series['original'][-window_len:])

            series['smooth'] = np.hstack([series['smooth'], smoother.smooth_data[:, -1]])

            _low, _up = smoother.get_intervals('sigma_interval', n_sigma=n_sigma)
            series['low'] = np.hstack([series['low'], _low[:, -1]])
            series['up'] = np.hstack([series['up'], _up[:, -1]])

            is_anomaly = np.logical_or(
                series['original'][-1] > series['up'][-1],
                series['original'][-1] < series['low'][-1]
            )

            if is_anomaly.any():
                series['idx'] = np.hstack([series['idx'], is_anomaly * i]).astype(int)

        if i >= timesteps:
            continue

        series['original'] = np.hstack([series['original'], vecX[i]])

    if len(series["idx"]) != 0:
        idx0 = np.where(series['original'] > threshold)[0]
        idx = np.intersect1d(idx0, series['idx'])
        if len(idx) != 0:
            if idx[-1] == len(vecX):
                idx[-1] = idx[-1] - 1
    y_qual = np.copy(vecX_qual)

    if len(idx) != 0:
        y_qual[idx] = 1
        y_qual = np.array(y_qual, dtype=bool)

    return y_qual


def qa_kmeans(vecX, ncluster=2):
    """
        Remove outliers based on kmean clustering.

        Parameters:
            vecX (np.array): Data array to which to apply the quality assurance
            ncluster (int) : number of cluster (>=2)
        Returns:
            y_qual (np.array): An array of bools where True means non-trusted data for this outlier detection
    """
    y_qual = np.zeros(len(vecX), dtype=bool)
    clusterer = KMeans(n_clusters=ncluster)
    clusterer.fit(vecX.reshape(-1, 1))

    nearest_centroid_idx = clusterer.predict(vecX.reshape(-1, 1))

    igr1 = np.where(nearest_centroid_idx == 0)[0]
    igr2 = np.where(nearest_centroid_idx == 1)[0]

    val_thresh = (np.mean(vecX[igr2]) - np.mean(vecX[igr1])) / np.quantile(vecX, 0.90)

    if val_thresh >= 5:  # if there is no 2 clearly seperated groups
        y_qual[igr2] = True

    y_qual = np.array(y_qual, dtype=bool)

    return y_qual


def qa_kmeans_threshold(vecX, ncluster=2, threshold=1.2):
    """
        Remove outliers based on kmean clustering and threshold value.

        Parameters:
            vecX (np.array): Data array to which to apply the quality assurance
            ncluster : number of cluster (>=2)
        Returns:
            y_qual (np.array): An array of bools where True means non-trusted data for this outlier detection
    """
    y_qual = np.zeros(len(vecX), dtype=bool)
    clusterer = KMeans(n_clusters=ncluster)
    clusterer.fit(vecX.reshape(-1, 1))

    nearest_centroid_idx = clusterer.predict(vecX.reshape(-1, 1))

    igr1 = np.where(nearest_centroid_idx == 0)[0]
    igr2 = np.where(nearest_centroid_idx == 1)[0]

    if len(igr1) > len(igr2):
        igrfin = igr2
    else:
        igrfin = igr1

    val_thresh = abs((np.mean(vecX[igr2]) - np.mean(vecX[igr1]))) / np.quantile(vecX, 0.90)

    if val_thresh < threshold:  # if there is no 2 clearly seperated groups
        igrfin = []
    else:
        y_qual[igrfin] = True

    return y_qual
