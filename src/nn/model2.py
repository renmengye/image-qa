from container2 import *

class GraphModel(GraphContainer):
    def __init__(self,
                 stages, 
                 outputStageNames,
                 costFn,
                 inputDim=0,
                 outputDim=0,
                 name=None,
                 decisionFn=None, 
                 specFilename=None):
        GraphContainer.__init__(self,
                 name=name,
                 stages=stages,
                 outputStageNames=outputStageNames,
                 inputDim=inputDim,
                 outputDim=outputDim,
                 inputType='float',
                 outputdEdX=True)
        self.getCost = costFn
        self.predict = decisionFn
        self.specFilename = specFilename