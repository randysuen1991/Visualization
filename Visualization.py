from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
import DimensionReductionApproaches as DRA
import numpy as np


class Visualization():
    def __init__(self,preprocess_components=None):
        self.visualizer = None
        self.preprocess_components = preprocess_components
        
    # Notice: the Y should be numbers qual to or bigger than 1
    def Fit(self,X,Y):
        if self.preprocess_components is not None:
            self.pca_subspace, _ = DRA.DimensionReduction.PCA(X_train=X, n_components=self.preprocess_components)
            self.preprocess_X = np.matmul(X,self.pca_subspace)
        else :
            self.preprocess_X = X
        self.visualizer.fit(self.preprocess_X)
        self.labels = Y 
        
    def Fit_Transform(self,X,Y):
        if self.preprocess_components != None :
            self.pca_subspace, _ = DRA.DimensionReduction.PCA(X_train=X, n_components=self.preprocess_components)
            self.preprocess_X = np.matmul(X,self.pca_subspace)
        else :
            self.preprocess_X = X
        self.X_2d = self.visualizer.fit_transform(self.preprocess_X)
        self.labels = Y 
        
    def Visualize(self):
        target_ids = range(int(max(self.labels.ravel())))
        for i in target_ids:
            print(self.labels == 1)
            plt.scatter(self.X_2d[(self.labels == i+1).ravel(), 0], self.X_2d[(self.labels == i+1).ravel(), 1], label=i+1)
        plt.legend()
        plt.show()


class tSNE(Visualization):
    def __init__(self,preprocess_components=None):
        super().__init__(preprocess_components)
        self.visualizer = TSNE()

        
# This approach is 
class MultiDimensionalScaling(Visualization):
    def __init__(self):
        pass



def test():
    import numpy as np

    from matplotlib import pyplot as plt
    from matplotlib.collections import LineCollection

    from sklearn import manifold
    from sklearn.metrics import euclidean_distances
    from sklearn.decomposition import PCA

    n_samples = 20
    seed = np.random.RandomState(seed=3)
    X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
    X_true = X_true.reshape((n_samples, 2))
    # Center the data
    X_true -= X_true.mean()

    similarities = euclidean_distances(X_true)
    print(similarities[0:3,0:3])
        
if __name__ == "__main__" :
    test()