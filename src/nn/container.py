from activation_fn import *
from fully_connected_layer import *

class Container(Layer):
    def __init__(self,
                 stages,
                 outputStageNames,
                 inputDim,
                 numNode,
                 inputNames,
                 name=None,
                 outputdEdX=True):
        Layer.__init__(self,
                       name=name,
                       numNode=numNode)
        self.stages = []
        self.stageDict = {}
        self.inputDim = inputDim
        self.outputStageNames = outputStageNames

    def graphForward(self):
        self._inputValue = self.getInput()
        self._outputValue = self.forward(self._inputValue)

    def forward(self, inputValue):
        pass

    def backward(self, gradientToOutput):
        pass