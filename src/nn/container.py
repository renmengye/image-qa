from stage import *

class Container(Stage):
    def __init__(self, stages, name=None, outputdEdX=True):
        Stage.__init__(self,
                 name=name,
                 outputdEdX=outputdEdX)
        self.stages = stages

    def forward(self, X, dropout=True):
        X1 = X
        for stage in self.stages:
            if isinstance(stage, Container):
                X1 = stage.forward(X1, dropout)
            elif hasattr(stage, 'dropout'):
                stage.dropout = dropout
                X1 = stage.forward(X1)
            else:
                X1 = stage.forward(X1)
        return X1

    def updateWeights(self):
        for stage in self.stages:
            stage.updateWeights()
        return

    def updateLearningParams(self, numEpoch):
        for stage in self.stages:
            stage.updateLearningParams(numEpoch)
        return

    def getWeights(self):
        weights = []
        for stage in self.stages:
            weights.append(stage.getWeights())
        return np.array(weights, dtype=object)

    def loadWeights(self, W):
        for i in range(W.shape[0]):
            self.stages[i].loadWeights(W[i])