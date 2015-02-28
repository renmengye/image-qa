from container import *

class Parallel(Container):
    """
    Pass the input two all substages, and concatenate the output.
    """
    def __init__(self, name, stages, axis, outputdEdX=True):
        Container.__init__(self, name=name, stages=stages, outputdEdX=outputdEdX)
        self.axis = axis

    def forward(self, X):
        self.splY = []
        for s in self.stages:
            self.splY.append(s.forward(X))
            #print self.splY[-1].shape
        return np.concatenate(self.splY, axis=self.axis)

    def backward(self, dEdY):
        start = 0
        for (s, Y) in zip(self.stages, self.splY):
            if self.axis == 0:
                s.backward(dEdY[start:start+Y.shape[0]])
                start += Y.shape[0]
            elif self.axis == 1:
                s.backward(dEdY[:,start:start+Y.shape[1]])
                start += Y.shape[1]