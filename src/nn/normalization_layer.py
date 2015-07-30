from layer import Layer
from environment import *

class NormalizationLayer(Layer):
    def __init__(self,
                 name,
                 mean,
                 std,
                 numNode,
                 gpuEnabled=USE_GPU):
        Layer.__init__(self,
                 name=name,
                 numNode=numNode,
                 gpuEnabled=gpuEnabled)
        self.mean = mean
        self.std = std

    def forward(self, inputValue):
        return (inputValue - self.mean) / self.std

    def backward(self, gradientToOutput):
        return gradientToOutput / self.std