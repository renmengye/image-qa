from container import *

class Concat(Container):
    """
    Split the input to different substages, and concatenate the output
    """
    def __init__(self, stages, axis, splits, name=None, outputdEdX=True):
        Container.__init__(self, stages=stages, name=name, outputdEdX=outputdEdX)
        self.axis = axis
        self.splits = splits
        self.outputSplits = []

    def forward(self, X):
        self.outputSplits = []
        splY = []
        splX = np.split(X, self.splits, axis=self.axis)
        lastIdx = 0
        for i in range(0, len(self.stages)):
            Ytmp = self.stages[i].forward(splX[i])
            splY.append(Ytmp)
            if i < len(self.stages) - 1:
                lastIdx = Ytmp.shape[self.axis] + lastIdx
                self.outputSplits.append(lastIdx)
        Y = np.concatenate(splY, axis=self.axis)
        return Y

    def backward(self, dEdY):
        spldEdX = []
        spldEdY = np.split(dEdY, self.outputSplits, axis=self.axis)
        for i in range(0, len(self.stages)):
            dEdXtmp = self.stages[i].backward(spldEdY[i])
            spldEdX.append(dEdXtmp)
        if self.outputdEdX:
            dEdX = np.concatenate(spldEdX, axis=self.axis)
            return dEdX
        else:
            return None