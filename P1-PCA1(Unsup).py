# Principal Component Analysis-dimensionality reduction
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.pyplot import scatter, show

def pca(n_component):
    cancer = load_breast_cancer()
    scaler = StandardScaler()
    scaler.fit(cancer.data)
    X_scaled = scaler.transform(cancer.data)
    for i in range(1, n_component):
        pca = PCA(n_components=i+1)
        pca.fit(X_scaled) # fit pca model to the data
        X_pca = pca.transform(X_scaled) # transform data onto the principal componentw
        print('original shape: {}'.format(str(X_scaled.shape)))
        print('original shape: {}'.format(str(X_pca.shape)))
        print('PCA component shape are: {}'.format(pca.components_.shape))
        scatter(X_pca[:,0], X_pca[:,1], c=(cancer.target+1)**3)
        show()

if __name__=='__main__':
    pca(int(input('Enter n_component:')))
