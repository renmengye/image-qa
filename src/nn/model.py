from stage import *

class Model(Stage):
    def __init__(self, stage, costFn, decisionFn=None, specFilename=None):
        self.stage = stage
        self.getCost = costFn
        self.predict = decisionFn
        self.specFilename = specFilename

    def forward(self, X, dropout=True):
        return self.stage.forward(X, dropout)

    def backward(self, dEdY):
        return self.stage.backward(dEdY)

    def updateWeights(self):
        return self.stage.updateWeights()

    def updateLearningParams(self, numEpoch):
        return self.stage.updateLearningParams(numEpoch)

    def getWeights(self):
        return self.stage.getWeights()

    def loadWeights(self, W):
        return self.stage.loadWeights(W)
