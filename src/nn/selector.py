from stage import *

class Selector(Stage):
    def __init__(self, 
                 name, 
                 inputNames,
                 start, 
                 end, 
                 axis=-1):
        Stage.__init__(
                 self,
                 name=name, 
                 inputNames=inputNames,
                 outputDim=end-start)
        self.start = start
        self.end = end
        self.axis = axis

    def forward(self, X):
        self.X = X
        if self.axis == 0:
            return X[self.start:self.end]
        else:
            return X[:, self.start:self.end]

    def backward(self, dEdY):
        dEdX = np.zeros(self.X.shape)
        if self.axis == 0:
            dEdX[self.start:self.end] = dEdY
        else:
            dEdX[:, self.start:self.end] = dEdY
        return dEdX