import numpy as np
from .methods import *
#from .methods import qa_numeric, qa_bounds, qa_edges, qa_iqr, maintenance_flag
from .functions import check_data


def qualityassurance(variable, time, **kwargs):
    """
        Quality assurance for timeseries data.

        Parameters:
            variable (np.array): Data array to which to apply the quality assurance
            time (np.array): Time array for the variable
        Returns:
            qa (np.array): An array of ints where > 0 means non-trusted data.
        """
    check_data(variable)
    qa = np.zeros(len(variable), dtype=int)

    if "numeric" in kwargs and kwargs["numeric"]:
        qa[qa_numeric(variable)] = 1

    if "bounds" in kwargs and kwargs["bounds"]:
        qa[qa_bounds(variable, kwargs["bounds"])] = 1

    if "edges" in kwargs and kwargs["edges"]:
        qa[qa_edges(variable, time, kwargs["edges"])] = 1

    if "IQR" in kwargs and kwargs["IQR"]:
        qa[qa_iqr(variable, kwargs["IQR"])] = 1

    if "variation_rate" in kwargs and kwargs["variation_rate"]:
        qa[qa_variation_rate(variable, kwargs["variation_rate"])] = 1
    
    if "IQR_moving" in kwargs and kwargs["IQR_moving"]:
        qa[qa_iqr_moving(variable, kwargs["IQR_moving"])] = 1

    if "IQR_window" in kwargs and kwargs["IQR_window"]:
        qa[qa_max(variable, kwargs["IQR_window"],qa)] = 1

    if "convolution" in kwargs and kwargs["convolution"]:
        qa[qa_convolution(variable, kwargs["convolution"], qa)] = 1 #used qa for testing the vecX_qual

    if "kmeans" in kwargs and kwargs["kmeans"]: 
        qa[qa_kmeans(time,kwargs["kmeans"])]=1
    
    if "kmeans_threshold" in kwargs and kwargs["kmeans_threshold"]:
        qa[qa_kmeans_threshold(variable, kwargs["kmeans_threshold"])] = 1
    
    return qa
