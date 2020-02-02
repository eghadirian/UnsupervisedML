# Manifold Learning Algorithm, t_SNE: mostly useful for gaining understanding as it won't work on new data
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import  TSNE
import matplotlib.pyplot as plt

digits=load_digits()
pca = PCA(n_components=2).fit(digits.data)
digits_pca = pca.transform(digits.data)
tsne = TSNE(random_state=42)
digits_tsne = tsne.fit_transform(digits.data)
for i in range(len(digits.data)):
    plt.text(digits_pca[i,0], digits_pca[i,1], str(digits.target[i]), fontdict={'weight':'bold', 'size':9})
plt.show()
for i in range(len(digits.data)):
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), fontdict={'weight':'bold', 'size':9})
plt.show()
