import sys
if 'C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches' not in sys.path :
    sys.path.append('C:\\Users\\ASUS\Dropbox\\pycode\\mine\\Dimension-Reduction-Approaches')
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import DimensionReductionApproaches as DRA
import numpy as np

class Visualization():
    def __init__(self,preprocess_components=None):
        self.visualizer = None
        self.preprocess_components = preprocess_components
        
    # Notice: the Y should be numbers qual to or bigger than 1
    def Fit(self,X,Y):
        if self.preprocess_components != None :
            self.pca_subspace, _ = DRA.DimensionReduction.PCA(X_train=X,n_components=self.preprocess_components)
            self.preprocess_X = np.matmul(X,self.pca_subspace)
        else :
            self.preprocess_X = X
        self.visualizer.fit(self.preprocess_X)
        self.labels = Y 
        
    def Fit_Transform(self,X,Y):
        if self.preprocess_components != None :
            self.pca_subspace, _ = DRA.DimensionReduction.PCA(X_train=X,n_components=self.preprocess_components)
            self.preprocess_X = np.matmul(X,self.pca_subspace)
        else :
            self.preprocess_X = X
        self.X_2d = self.visualizer.fit_transform(self.preprocess_X)
        self.labels = Y 
        
    def Visualize(self):
        target_ids = range(int(max(self.labels.ravel())))
        for i in target_ids:
            print(self.labels==1)
            plt.scatter(self.X_2d[(self.labels == i+1).ravel(), 0], self.X_2d[(self.labels == i+1).ravel(), 1], label=i+1)
        plt.legend()
        plt.show()
    
class tSNE(Visualization):
    def __init__(self,preprocess_components=None):
        super().__init__(preprocess_components)
        self.visualizer = TSNE()

        


def test():
    from sklearn import datasets
    digits = datasets.load_digits()
    # Take the first 500 data points: it's hard to see 1500 points
    X = digits.data[:500]
    y = digits.target[:500]
    target_ids = range(len(digits.target_names))

    print(digits.target_names,target_ids)

        
if __name__ == "__main__" :
    test()