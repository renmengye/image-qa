from container import *

class Parallel(Container):
    """
    Pass the input two all substages, and concatenate the output.
    """
    def __init__(self, name, stages, axis, outputdEdX=True):
        Container.__init__(self, name=name, stages=stages, outputdEdX=outputdEdX)
        self.axis = axis
        self.splY = None

    def forward(self, X, dropout=True):
        self.splY = []
        for stage in self.stages:
            if isinstance(stage, Container):
                Ytmp = stage.forward(X, dropout)
            elif hasattr(stage, 'dropout'):
                stage.dropout = dropout
                Ytmp = stage.forward(X)
            else:
                Ytmp = stage.forward(X)
            self.splY.append(Ytmp)
        return np.concatenate(self.splY, axis=self.axis)

    def backward(self, dEdY):
        start = 0
        dEdX = 0.0
        for (s, Y) in zip(self.stages, self.splY):
            if self.axis == 0:
                dEdXTmp = s.backward(dEdY[start:start+Y.shape[0]])
                start += Y.shape[0]
            elif self.axis == 1:
                dEdXTmp = s.backward(dEdY[:,start:start+Y.shape[1]])
                start += Y.shape[1]
            if self.outputdEdX:
                dEdX += dEdXTmp
        return dEdX if self.outputdEdX else None