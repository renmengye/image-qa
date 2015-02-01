from stage import *

class Parallel(Stage):
    def __init__(self, stages, axis, splits):
        Stage.__init__(self)
        self.stages = stages
        self.axis = axis
        self.splits = splits
        self.outputSplits = []

    def forwardPass(self, X):
        self.outputSplits = []
        splY = []
        splX = np.split(X, self.splits, axis=self.axis)
        lastIdx = 0
        for i in range(0, len(self.stages)):
            Ytmp = self.stages[i].forwardPass(splX[i])
            splY.append(Ytmp)
            self.outputSplits.append(Ytmp.shape[self.axis] + lastIdx)
        Y = np.concatenate(splY, axis=self.axis)
        return Y

    def backPropagate(self, dEdY):
        spldEdX = []
        spldEdY = np.split(dEdY, self.outputSplits, axis=self.axis)
        for i in range(0, len(self.stages)):
            dEdXtmp = self.stages[i].backPropagate(spldEdY[i])
            spldEdX.append(dEdXtmp)
        dEdX = np.concatenate(spldEdX, axis=self.axis)
        return dEdX if self.outputdEdX else None

    def updateWeights(self):
        for i in range(0, len(self.stages)):
            self.stages[i].updateWeights()
        return
