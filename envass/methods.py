import numpy as np
from .functions import isnt_number, interp_nan, init_flag, nan_helper
import pandas as pd
from sklearn.cluster import KMeans
from datetime import datetime


def qa_numeric(variable, prior_flags=False):
    """
    Indicate values that are not considered numerical 

    Parameters:
        variable (np.array): Data array to which to apply the quality assurance
    Returns:
        flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
    """
    
    flags = init_flag(variable, prior_flags)
    isnt_numeric = np.vectorize(isnt_number, otypes=[bool])
    flags[isnt_numeric(variable)] = True
    return flags

def qa_bounds(variable, bounds, prior_flags=False):
    """¨
    Indicate values which are not in the range specified by the bounds

    Parameters:
        variable (np.array): Data array to which to apply the quality assurance
    Returns:
        flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
    """
    data = np.squeeze(np.array(pd.DataFrame(variable).apply(pd.to_numeric, errors='coerce').astype(np.float)))
    data[qa_numeric(data)] = np.nan
    flags = init_flag(variable, prior_flags)
    flags[~np.isnan(data)] = np.logical_or(data[~np.isnan(data)] < float(bounds[0]), data[~np.isnan(data)] > float(bounds[1]))
    return flags

def qa_edges(variable, time, edges, prior_flags=False):
    """
    Indicate values on the edges of the data set

    Parameters:
    variable (np.array): Data array to which to apply the quality assurance
        time (np.array): Time array corresponding to the Data array, time should be in seconds
        edges (int): time (s) in which the data will be cutted on the edges
        prior_flags (np.array): An array of bools where True means non-trusted data
    Returns:
        flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
    """
    flags = np.atleast_2d(init_flag(variable, prior_flags))
    flags[:,time > time[-1] - edges] = True
    flags[:,time < time[0] + edges] = True
    return np.squeeze(flags)

def qa_monotonic(time, monotonic, prior_flags=False):
    '''
    Indicate values which are not (strictly) increasing/decreasing

    Parameters:
        time (np.array): Time array corresponding to the Data array, time should be in seconds
        monotonic (str): Indicate if the vector should be: strictly_increasing, increasing, strictly decreasing or decreasing
        prior_flags (np.array): An array of bools where True means non-trusted data
    Returns:
        flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
    '''
    flags = init_flag(time, prior_flags)
    if 'strictly_increasing'in monotonic:
        flags[np.where(np.diff(time)<=0)[0]+1] = 1
    if 'strictly_decreasing'in monotonic:
        flags[np.where(np.diff(time)>=0)[0]+1] = 1
    if 'increasing'in monotonic:
        flags[np.where(np.diff(time)<0)[0]+1] = 1
    if 'decreasing'in monotonic:
        flags[np.where(np.diff(time)>0)[0]+1] = 1
    return flags

def qa_iqr(variable, time, factor=3, prior_flags=False):
    """
    Indicate values outside interquartile Range (IQR) for timeseries data.

    Parameters:
        variable (np.array): Data array to which to apply the quality assurance
        factor (int): threshold for outlier labelling rule
        prior_flags (np.array): An array of bools where True means non-trusted data

    Returns:
        flag (np.array): An array of bools where True means non-trusted data after this outlier detection
    """
    data = interp_nan(time, np.copy(variable))
    flags = init_flag(time, prior_flags)

    q75 = np.quantile(data, 0.75)
    q25 = np.quantile(data, 0.25)
    iqr = q75 - q25
    if iqr != 0:
        sup = q75 + factor * iqr
        inf = q25 - factor * iqr
        idx1 = np.where(data >= sup)[0]
        idx2 = np.where(data <= inf)[0]
        idx0 = np.r_[idx1, idx2]
    else:
        idx0 = np.array([])

    flags[idx0] = True
    return flags

def qa_variation_rate(variable, time, prior_flags=False):
    """
    Indicate the trustability of the  values if variation rate exceed a defined threshold.

    Parameters:
        variable (np.array): Data array to which to apply the quality assurance 
        time (np.array): Time array corresponding to the Data array  
        prior_flags (np.array): An array of bools where True means non-trusted data
    Returns:
        flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
    """
    
    data = interp_nan(time, np.copy(variable))
    flags = init_flag(time, prior_flags)

    vec_diff = abs(np.diff(data))
    if len(vec_diff) > 0:
        vecdiff_quan = np.quantile(vec_diff, 0.999)
        idx_vecdiff = np.where(vec_diff >= vecdiff_quan)[0]

        vec_max = np.max(data)
        if vec_max / np.quantile(data, 0.99) < 2:
            quantile_threshold = 0.99
        elif vec_max / np.quantile(data, 0.999) < 2:
            quantile_threshold = 0.999
        elif vec_max / np.quantile(data, 0.9999) < 2:
            quantile_threshold = 0.9999
        elif vec_max / np.quantile(data, 0.99999) < 2:
            quantile_threshold = 0.99999

        vec_quan = np.quantile(data, quantile_threshold)
        idx_vec = np.where(data >= vec_quan)[0]
        idx = list(set(idx_vecdiff) & set(idx_vec)) 
    else:
        idx = []
    flags = np.array(flags, dtype=bool)
    flags[idx] = True
    return flags

def qa_iqr_moving(variable, time, window_size=15, factor=3, prior_flags=False):
    """
   Indicate outliers values based on Interquartile Range (IQR) for a window of time series data

   Parameters:
       variable (np.array): Data array to which to apply the quality assurance
       windowsize (np.int): window size of data
       prior_flags (np.array): An array of bools where True means non-trusted data

   Returns:
       flags (np.array): An array of bools where True means non-trusted data for this outlier dectection
   """
    data = interp_nan(time, np.copy(variable))

    flags = init_flag(time, prior_flags)

    if len(data) < window_size:
        print("ERROR! Window size is larger than array length.")
    else:
        for i in range(0, (len(data) - window_size + 1)):
            data_sub = np.copy(data[i:i + window_size])
            q75 = np.quantile(data_sub, 0.75)
            q25 = np.quantile(data_sub, 0.25)
            IQR = q75 - q25
            outsup = q75 + factor * IQR
            outinf = q25 - factor * IQR

            idx1 = np.where(data_sub >= outsup)[0]
            idx2 = np.where(data_sub <= outinf)[0]

            idx0 = np.r_[idx1, idx2]

            if len(idx0) != 0:
                flags[i + idx0] = flags[i + idx0] + 1
    flags[flags>1]=1
    flags = np.array(flags, dtype=bool)
    return flags

def qa_max(variable, time, factor=3, semiwindow=1000, prior_flags=False):
    """
        Indicate outliers values based on Interquartile Range (IQR) for a window of time series data

       Parameters:
           variable (np.array): Data array to which to apply the quality assurance
           semiwindow (int): window size of data
           factor (int): threshold for outlier labelling rule
           prior_flags (np.array): An array of bools where True means non-trusted data

       Returns:
           flags (np.array): An array of bools where True means non-trusted data for this outlier detection
   """
    data = interp_nan(time, np.copy(variable))
    flags = init_flag(time, prior_flags)

    maxy = np.nanmax(data)

    n0 = np.where(data == maxy)[0][0] - semiwindow
    n1 = np.where(data == maxy)[0][0] + semiwindow

    if n0 < 0:
        n0 = 0  # maybe not ! check for the last dataset
    if n1 > (len(data) - 1):
        n1 = len(data) - 1 #Yes but add the non evaluated dataset to the next bin

    vec_sub = data[n0:n1]
    vec_qual = np.zeros(len(vec_sub))
    vec_time = time[n0:n1]
    flags99 = qa_iqr(vec_sub,vec_time,factor)
    if sum(flags99) > 0:
        flags[n0:n1] = flags[n0:n1] + flags99  # update vec_qual

    flags = np.array(flags, dtype=bool)
    return flags

def qa_kmeans(variable, time, ncluster=2, prior_flags = False):
    """
        Indicate outliers based on kmean clustering.

        Parameters:
            variable (np.array): Data array to which to apply the quality assurance
            ncluster (int) : number of cluster (>=2)
            prior_flags (np.array): An array of bools where True means non-trusted data
        Returns:
            flags (np.array): An array of bools where True means non-trusted data for this outlier detection
    """
    data = interp_nan(time, np.copy(variable))
    flags = init_flag(time, prior_flags)

    clusterer = KMeans(n_clusters=ncluster)
    clusterer.fit(data.reshape(-1, 1))

    nearest_centroid_idx = clusterer.predict(data.reshape(-1, 1))

    igr1 = np.where(nearest_centroid_idx == 0)[0]
    igr2 = np.where(nearest_centroid_idx == 1)[0]

    val_thresh = (np.mean(data[igr2]) - np.mean(data[igr1])) / np.quantile(data, 0.90)

    if val_thresh >= 5:  # if there is no 2 clearly seperated groups
        flags[igr2] = True

    flags = np.array(flags, dtype=bool)

    return flags

def qa_kmeans_threshold(variable, time, ncluster=2, threshold=1.2, prior_flags=False):
    """
        Indicate outliers based on kmean clustering and threshold value.

        Parameters:
            variable (np.array): Data array to which to apply the quality assurance
            ncluster : number of cluster (>=2)
            prior_flags (np.array): An array of bools where True means non-trusted data
        Returns:
            flags (np.array): An array of bools where True means non-trusted data for this outlier detection
    """
    data = interp_nan(time, np.copy(variable))
    flags = init_flag(time, prior_flags)

    clusterer = KMeans(n_clusters=ncluster)
    clusterer.fit(data.reshape(-1, 1))

    nearest_centroid_idx = clusterer.predict(data.reshape(-1, 1))

    igr1 = np.where(nearest_centroid_idx == 0)[0]
    igr2 = np.where(nearest_centroid_idx == 1)[0]

    if len(igr1) > len(igr2):
        igrfin = igr2
    else:
        igrfin = igr1

    val_thresh = abs((np.mean(data[igr2]) - np.mean(data[igr1]))) / np.quantile(data, 0.90)

    if val_thresh < threshold:  # if there is no 2 clearly seperated groups
        igrfin = []
    else:
        flags[igrfin] = True
    flags = np.array(flags, dtype=bool)
    return flags

def qa_maintenance(time,path='maintenance_log.csv', prior_flags=False):
    """
        Indicate the trustability of values based on the maintenance logbook

        Parameters:
            time (np.array): Time array to which to apply the quality assurance
            prior_flags (np.array): An array of bools where True means non-trusted data
            additional comments: The maintenance_log.csv should have a date format '%Y-%m-%d %H:%M:%S.%f'
        Returns:
            flags (np.array): An array of bools where True means non-trusted data
    """
    maintenance_log=pd.read_csv(path, sep=';')

    flags = init_flag(time, prior_flags)
    
    start=maintenance_log.start.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    end=maintenance_log.end.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f'))
    maintenance_end=[datetime.timestamp(s) for s in end]
    maintenance_start=[datetime.timestamp(s) for s in start]
    
    mask=[]
    for i in  range(0,len(maintenance_end)):
        mask=(time>maintenance_start[i]) & (time<maintenance_end[i])
        flags[mask]=True

    return flags


def qa_individual(time, individual_check, prior_flags = False):
    """ 
        Read individual checks and flag 
        
        Parameters: 
            time (np.array): Time array to which to apply the quality assurance
            individual_check (list): List with points in time having to be removed 
            prior_flags (np.array): An array of bools where True means non-trusted data
        Returns:
            flags (np.array): An array of bools where True means non-trusted data
            """
    flags = init_flag(time, prior_flags)
    for i in individual_check:
        flag_idx = np.where(i==time)[0]
        flags[flag_idx] = True
    return flags


def qa_moving_average_limit(variable, window=100, limit=5, prior_flags=False):
    """
    Indicate values on the edges of the data set
    Parameters:
        variable (np.array): Data array to which to apply the quality assurance
        window (int): Size of window for moving average
        limit (flaot): Allowable distance from the moving average
        prior_flags (np.array): An array of bools where True means non-trusted data
    Returns:
        flag (np.array): An array of bools where True means non-trusted data for this outlier dectection
    """
    flags = init_flag(variable, prior_flags)
    ma = np.convolve(variable, np.ones(window), 'same') / window
    nans, xx = nan_helper(ma)
    ma[nans] = np.interp(xx(nans), xx(~nans), ma[~nans])
    flags[np.abs(ma - variable) > limit] = True
    return flags

