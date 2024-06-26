import numpy as np
from . import methods
from . import functions


def qualityassurance(variable, time, **kwargs):
    """
        Quality assurance for timeseries data.

        Parameters:
            variable (np.array): Data array to which to apply the quality assurance
            time (np.array): Time array for the variable
            kwargs (dictionary): Integrate the type of test to perform with his corresponding parameters
            type of test supported: numeric, bounds, edges, IQR, variation_rate, IQR_moving, IQR_window, convolution, kmeans, kmeans_threshold, maintenance, 
            parameters examples: window_size, window_type,semiwindow, ncluster, threshold
            e.g. {"numeric":True,"IQR":{"factor":3}}
        Returns:
            qa (np.array): An array of ints where > 0 means non-trusted data.
        """
    
    functions.check_data(variable, time)
    qa = np.zeros(variable.shape, dtype=int)
    kwargs = functions.to_dict(kwargs)

    if "numeric" in kwargs:
        qa[methods.qa_numeric(variable)] = 1

    if "bounds" in kwargs:
        qa[methods.qa_bounds(variable, **kwargs["bounds"])] = 1

    if "edges" in kwargs:
        qa[methods.qa_edges(variable, time, **kwargs["edges"])] = 1

    if "monotonic" in kwargs:
        qa[methods.qa_monotonic(time, **kwargs["monotonic"])] = 1
        
    if "IQR" in kwargs:
        qa[methods.qa_iqr(variable, time, **kwargs["IQR"])] = 1

    if "variation_rate" in kwargs:
        qa[methods.qa_variation_rate(variable, time, **kwargs["variation_rate"])] = 1
    
    if "IQR_moving" in kwargs:
        qa[methods.qa_iqr_moving(variable, time, **kwargs["IQR_moving"])] = 1

    if "IQR_window" in kwargs:
        qa[methods.qa_max(variable, time, **kwargs["IQR_window"])] = 1

    if "kmeans" in kwargs:
        qa[methods.qa_kmeans(variable, time, **kwargs["kmeans"])] = 1
    
    if "kmeans_threshold" in kwargs:
        qa[methods.qa_kmeans_threshold(variable, time, **kwargs["kmeans_threshold"])] = 1
    
    if "maintenance" in kwargs:
        qa[methods.qa_maintenance(time, **kwargs["maintenance"])] = 1

    if "individual_check" in kwargs:
        qa[methods.qa_individual(time, **kwargs["individual_check"])] = 1

    return qa

