import numpy as np
from .context import envass


def test_qualityassurance():
    variable = np.array([1, "g", 16, 12.0, False, 0, 22.12, 5.77])
    correct_output = np.array([1, 1, 0, 0, 1, 1, 1, 0])
    time = np.array(range(len(variable)))
    output = envass.qualityassurance(variable, time, numeric=True, bounds=[5, 21])
    assert np.array_equal(correct_output, output)
