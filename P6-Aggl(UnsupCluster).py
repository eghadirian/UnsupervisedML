# Agglomerate clustering: hierarchical clustering
# cannot make prediction for new data
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, ward
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=1)
agg = AgglomerativeClustering(n_clusters=3)
assignment = agg.fit_predict(X)
linkage_array = ward(X)
plt.scatter(X[:,0], X[:,1], assignment, marker='o', c=assignment)
plt.show()
dendrogram(linkage_array)
ax = plt.gca()
plt.show()