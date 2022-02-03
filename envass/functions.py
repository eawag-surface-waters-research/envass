import numpy as np


def check_data(x):
    if type(x).__module__ != np.__name__:
        raise TypeError("Input must be a numpy array.")


def isnt_number(n):
    try:
        float(n)
    except ValueError:
        return True
    else:
        return False