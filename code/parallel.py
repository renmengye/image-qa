from stage import *

class Parallel(Stage):
    def __init__(self, stages, axis, splits, name=None, outputdEdX=True):
        Stage.__init__(self, name=name, outputdEdX=outputdEdX)
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
            if i < len(self.stages) - 1:
                lastIdx = Ytmp.shape[self.axis] + lastIdx
                self.outputSplits.append(lastIdx)
        Y = np.concatenate(splY, axis=self.axis)
        return Y

    def backPropagate(self, dEdY):
        spldEdX = []
        spldEdY = np.split(dEdY, self.outputSplits, axis=self.axis)
        for i in range(0, len(self.stages)):
            dEdXtmp = self.stages[i].backPropagate(spldEdY[i])
            spldEdX.append(dEdXtmp)
        if self.outputdEdX:
            dEdX = np.concatenate(spldEdX, axis=self.axis)
            return dEdX
        else:
            return None

    def updateWeights(self):
        for i in range(0, len(self.stages)):
            self.stages[i].updateWeights()
        return

    def getWeights(self):
        weights = []
        for stage in self.stages:
            weights.append(stage.getWeights())
        return np.array(weights, dtype=object)

    def loadWeights(self, W):
        for i in range(W.shape[0]):
            self.stages[i].loadWeights(W[i])
        return
