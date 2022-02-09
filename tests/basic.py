import numpy as np
#from envass import qualityassurance
from qualityassurance.main import qualityassurance #Local package

variable = np.array([1, "g", 16, 12.0, False, 0, 22.12, 5.77])
time = np.array(range(len(variable)))

output = qualityassurance(variable, time, numeric=True, bounds=[5, 21])

print(variable)
print(output)
