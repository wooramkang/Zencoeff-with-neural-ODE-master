import numpy as np
import matplotlib.pyplot as plt

poi_dist = [0 for i in range(1000)]

axis_x = [ i/1000 for i in range(1000)]
poi_dist = np.array(poi_dist)
poi_dist = poi_dist - min(poi_dist)
poi_dist = poi_dist / (max(poi_dist) - min(poi_dist))
plt.figure()
plt.plot(axis_x[0:50], poi_dist[0:50])
#plt.hist(s, 14, density=True)
plt.show()
plt.savefig("pis.png")