from sklearn.manifold import TSNE



class Visualization():
    def __init__(self):
        self.visualizer = None
        
    def Fit(self,X):
        self.visualizer.fit(X)
        
    def Fit_Transform(self,X):
        self.visualizer.fit_transform(X)
        
    def Visualize(self):
        pass
    
    
def tSNE():
    def __init__(self):
        self.visualizer = TSNE()
    def Fit(self,X):
        self.visualizer.fit(X)
    def Fit_Transform(self,X):
        self.visualizer.fit_transform(X)
        
    
        
        