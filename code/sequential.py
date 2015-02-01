from stage import *

class Sequential(Stage):
    def __init__(self, stages):
        Stage.__init__(self)
        self.stages = stages

    def forwardPass(self, X, dropout=True):
        X1 = X
        for stage in self.stages:
            if hasattr(stage, 'dropout'):
                stage.dropout = dropout
                X1 = stage.forwardPass(X1)
            else:
                X1 = stage.forwardPass(X1)
        return X1

    def backPropagate(self, dEdY):
        for stage in reversed(self.stages):
            dEdY = stage.backPropagate(dEdY)
            if dEdY is None: break
        return dEdY

    def updateWeights(self):
        for stage in self.stages:
            stage.updateWeights()
        return

    def updateLearningParams(self, numEpoch):
        for stage in self.stages:
            stage.updateLearningParams(numEpoch)
        return