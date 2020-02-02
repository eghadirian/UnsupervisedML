# Non-negative Matrix Factorization
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt

people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
image_shape = people.images[0].shape
mask=np.zeros(people.target.shape, dtype=bool)
for target in np.unique(people.target):
    mask[np.where(people.target==target)[0][:50]]=1
X_people=people.data[mask]/255. # scaling from 0-255 to 0-1
y_people=people.target[mask]
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
nmf = NMF(n_components=15, random_state=0)
nmf.fit(X_train)
X_train_nmf = nmf.transform(X_train)
X_test_nmf = nmf.transform(X_test)
_, axes = plt.subplots(3,5, figsize=(15,12),subplot_kw={'xticks':(), 'yticks':()})
for i , (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
    ax.imshow(component.reshape(image_shape))
    ax.set_title('{}. component'.format(i))
plt.show()