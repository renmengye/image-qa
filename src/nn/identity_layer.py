from layer import *

class IdentityLayer(Layer):
    def __init__(self, name):
        Layer.__init__(self, name=name, useGpu=False)
        pass

    def graphForward(self):
        pass

    def backward(self, gradientToOutput):
        return gradientToOutput