from stage import *

class Reshape(Stage):
    def __init__(self, reshapeFn, inputNames=None, outputDim=0, name=None):
        Stage.__init__(self, name=name, inputNames=inputNames, outputDim=outputDim)
        self.reshapeFn = eval('lambda x: ' + reshapeFn)
        self.Xshape = 0

    def forward(self, X):
        self.Xshape = X.shape
        return np.reshape(X, self.reshapeFn(X.shape))

    def backward(self, dEdY):
        return np.reshape(dEdY, self.Xshape)

class TimeUnfold(Reshape):
    def __init__(self, name=None):
        Reshape.__init__(self, name=name, 
            reshapeFn='(x[0] * x[1], x[2])')

class TimeFold(Reshape):
    def __init__(self, timespan, name=None):
        self.timespan = timespan
        t = str(self.timespan)
        Reshape.__init__(self, name=name, 
            reshapeFn='(x[0] / '+t+','+t+', x[1])')