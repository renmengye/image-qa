from stage import *

class ComponentProduct(Stage):
    def __init__(self, name):
        Stage.__init__(
            self,
            name=name)
    def forward(self, X):
        self.X = X
        self.Y = X[:,:X.shape[1]/2] * X[:,X.shape[1]/2:]
        return self.Y
    def backward(self, dEdY):
        self.dEdW = 0.0
        return np.concatenate((self.X[:,self.X.shape[1]/2:] * dEdY, self.X[:,:self.X.shape[1]/2] * dEdY), axis=-1)
