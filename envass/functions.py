import numpy as np
import warnings

def check_data(x, time):
    if type(x).__module__ != np.__name__:
        raise TypeError("Input must be a numpy array.")
    if len(x)!=len(time):
        if len(time) in x.shape:
            print("2D array recognized")
            if x.shape[0]>x.shape[1]:
                warnings.warn("Numbers of rows is greater than numbers of columns, rows must be depth and columns is time !")
        else: 
            raise TypeError("Time and variable data should be of the same length")

def to_dict(kwargs):
    for key in list(kwargs):
        if type(kwargs[key])!=dict:
            if kwargs[key]==False:
                kwargs.pop(key)
            else:
                kwargs[key]={key:kwargs[key]}
    return kwargs

def check_parameters(test, kwargs, parameters):
    for arguments in kwargs[test].keys():
        if arguments not in parameters[test]:
            
            warnings.warn(f"argument {arguments} is not given to the function")

def isnt_number(n):
    try:
        if np.isnan(float(n)):
            return True
    except ValueError:
        return True
    else:
        return False

def init_flag(time, prior_flags):
    try: 
        if len(prior_flags):
            flag = np.array(np.copy(prior_flags),dtype=bool)
    except:
        flag = np.zeros(len(time), dtype=bool)
    return flag


def interp_nan(time, y):
    vec=np.copy(y)
    nans, x = np.isnan(vec), lambda z: z.nonzero()[0]
    vec[nans]=np.interp(time[nans], time[~nans],vec[~nans])
    return vec