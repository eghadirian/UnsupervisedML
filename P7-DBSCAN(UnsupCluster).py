# DBSCAN: density-based spatial clustering of applications with noise
# does not allow clustering of new points
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=1)
dbs = DBSCAN(min_samples=2, eps=1)
assignment = dbs.fit_predict(X)
plt.scatter(X[:,0], X[:,1], assignment, marker='o', c=assignment)
plt.show()
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
print('ARI for random: {}, for DBSCAN: {}'.format(adjusted_rand_score(y, random_clusters),
                                                  adjusted_rand_score(y, assignment)))
