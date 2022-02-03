import numpy as np
from .methods import qa_numeric, qa_bounds, qa_edges
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

    return qa
