# from layer import *
#
# class ConstantLayer(Layer):
#     def __init__(self,
#                  name,
#                  inputNames,
#                  outputDim,
#                  value):
#         Layer.__init__(self,
#                  name=name,
#                  outputDim=outputDim,
#                  inputNames=inputNames,
#                  outputdEdX=False)
#         self.dEdW = 0
#         self.value = value
#
#     def graphBackward(self):
#         self.backward(self._gradientToOutput)
#
#     def forward(self, inputValue):
#         return np.zeros((inputValue.shape[0], self.outputDim)) + self.value
#
#     def backward(self, gradientToOutput):
#         return None