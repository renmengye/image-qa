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
    def __init__(self, inputNames=None, name=None):
        Reshape.__init__(self,
                         name=name,
                         inputNames=inputNames,
                         reshapeFn='(x[0] * x[1], x[2])')

class TimeFold(Reshape):
    def __init__(self, timespan, inputNames=None, name=None):
        self.timespan = timespan
        t = str(self.timespan)
        Reshape.__init__(self,
                         name=name,
                         inputNames=inputNames,
                         reshapeFn='(x[0] / '+t+','+t+', x[1])')

class TimeConcat(Stage):
    def __init__(self, inputNames=None, name=None):
        Stage.__init__(self, name=name, inputNames=inputNames, outputDim=0)
    def getInput(self):
        if len(self.inputs) > 1:
            self.splX = []
            for stage in self.inputs:
                X = stage.Y
                self.splX.append(X)
            return np.concatenate(self.splX, axis=1)
        else:
            return self.inputs[0].Y
    def sendError(self, dEdX):
        """
        Iterates over input list and sends dEdX.
        """
        if len(self.inputs) > 1:
            s = 0
            for stage in self.inputs:
                s2 = s + stage.Y.shape[1]
                stage.dEdY += dEdX[:, s : s2]
                s = s2
        else:
            self.inputs[0].dEdY += dEdX

    def forward(self, X):
        return X
    def backward(self, dEdY):
        return dEdY