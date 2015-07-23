from layer import *

class IdentityLayer(Layer):
    def __init__(self, name, numNode):
        Layer.__init__(self, name=name, numNode=numNode, gpuEnabled=USE_GPU)
        pass

    def graphForward(self):
        pass

    def backward(self, gradientToOutput):
        return gradientToOutput