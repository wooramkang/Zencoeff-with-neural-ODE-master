import numpy as np
import matplotlib.pyplot as plt
s = np.random.poisson(16, 3125)
print(s)
plt.hist(s, 14, density=True)
plt.show()
plt.savefig("pis.png")