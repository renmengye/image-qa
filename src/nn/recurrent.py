class InputSub(RecurrentSubstage):
    # Just forward X = X
    # All the recurrent stages need to be symbolic.
    # For now, we can assume that the order of the stages listed in the model follows the dependency
    # so we don't have to worry about the dependency.

class RecurrentSubstage(Stage):
    def __init__(self):
        self.inputs = []
        self.dEdY = 0.0
        pass

    def getValue(self):
        pass

    def getInput(self):
        # fetch input from each input stage
        # concatenate input into one vector
        pass

    def receiveError(self, error):
        self.dEdY = error
        pass

    def sendError(self):
        # iterate over input list and send dEdX

    def forwardR(self):
        return self.forward(self.getInput())

    def backwardR(self):
        dEdX = self.backward(self.error)
        self.notifyError(dEdX)
        return dEdX

    def forward(self, X):
        """Subclasses need to implement this"""
        pass

    def backward(self, dEdY):
        """Subclasses need to implement this"""
        pass

    def updateWeights(self):
        """Update weights is prohibited here"""
        raise Exception('Weights update not allowed in recurrent substages.')
        pass

class Recurrent(Stage):
    """
    Recurrent container.
    Propagate through time.
    """
    def __init__(self, stages, name=None, outputdEdX=True):
        pass

    def forward(self, X):
        for t in range(X.shape[1]):
            for stage in self.stages:
                stage.forward()
        pass

    def backward(self, dEdY):
        pass

    def updateWeights(self):
        pass

    def updateLearningParams(self, numEpoch):
        pass