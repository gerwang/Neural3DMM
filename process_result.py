import json

import numpy as np
from matplotlib import pyplot

data = json.load(open('result.json'))
data = [(float(key), value) for key, value in data.items()]
data = np.array(sorted(data))

fig, ax = pyplot.subplots()
ax.set_xscale("log")
ax.plot(data[:, 0], data[:, 1])
fig.show()
