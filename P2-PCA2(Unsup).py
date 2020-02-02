# Principal Component Analysis - feature extraction
from sklearn.datasets import fetch_lfw_people
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def pcs_feature_extraction(n):
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    mask=np.zeros(people.target.shape, dtype=bool)
    for target in np.unique(people.target):
        mask[np.where(people.target==target)[0][:50]]=1
    X_people=people.data[mask]/255. # scaling from 0-255 to 0-1
    y_people=people.target[mask]
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
    print('KNN classifier test score: {}'.format(knn.score(X_test, y_test)))
    x=[]
    for i in range(1,n):
        pca = PCA(n_components=i+1, whiten=True, random_state=0).fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_pca, y_train)
        x.append(knn.score(X_test_pca, y_test))
    print('KNN classifier test score after the {} PCA: {}'.format(np.argmax(x)+1,x[np.argmax(x)]))
    pca = PCA(n_components=np.argmax(x) + 1, whiten=True, random_state=0).fit(X_train)
    _, axes = plt.subplots(3,5,figsize=(15,12), subplot_kw={'xticks':(), 'yticks':()})
    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title('{}.component'.format(i+1))
    plt.show()

if __name__=='__main__':
    pcs_feature_extraction(int(input('Enter n_component:')))