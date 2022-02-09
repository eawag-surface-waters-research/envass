# ENVironmental data quality ASSurance

ENVironmental data quality ASSurance for generating high quality data products.

## Installation

`pip install envass`

## Usage

```python

import numpy as np
from envass import qualityassurance

variable = np.array([1, "g", 16, 12.0, False, 0, 22.12, 5.77])
time = np.array(range(len(variable)))
checks={"numeric":{}, "IQR":{"factor":4}, "IQR_window":{}}

qa = qualityassurance(variable, time, **checks)
```
