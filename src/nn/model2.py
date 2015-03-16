from container2 import *

class GraphModel(GraphContainer):
    def __init__(self, 
                 stages, 
                 outputStageNames, 
                 inputDim, 
                 outputDim, 
                 costFn, 
                 decisionFn=None, 
                 specFilename=None):
        GraphContainer.__init__(self,
                 stages=stages,
                 outputStageNames,
                 inputDim=inputDim,
                 outputDim=outputDim,
                 inputType='float',
                 multiOutput=True,
                 name=None,
                 outputdEdX=True):
        self.getCost = costFn
        self.predict = decisionFn
        self.specFilename = specFilename