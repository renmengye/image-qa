import numpy as np

class GradientChecker():
    def __init__(self, layer, tolerance=1e-4, epsilon=1e-5):
        self._tolerance = tolerance
        self._epsilon = epsilon
        self._layer = layer

    def computeGradientToInput(self, inputValue):
        outputValue = self._layer.forward(inputValue)
        gradientToOutput = -outputValue
        gradient = self._layer.backward(gradientToOutput)
        gradientNumerical = np.zeros(gradient.size)
        inputValueReshape = inputValue.reshape(inputValue.size)
        for i in range(inputValueReshape.size):
            inputValueReshape[i] += self._epsilon
            inputValueTmp = inputValueReshape.reshape(inputValue.shape)
            outputValueTmpPlus = self._layer.forward(inputValueTmp)
            lossPlus = .5 * np.sum(outputValueTmpPlus ** 2)
            inputValueReshape[i] -= 2 * self._epsilon
            inputValueTmp = inputValueReshape.reshape(inputValue.shape)
            outputValueTmpMinus = self._layer.forward(inputValueTmp)
            lossMinus = .5 * np.sum(outputValueTmpMinus ** 2)
            gradientNumerical[i] = (lossPlus - lossMinus)\
                                   / 2 / self._epsilon
        return gradient, gradientNumerical.reshape(gradient.shape)

    def computeGradientToWeight(self, inputValue):
        outputValue = self._layer.forward(inputValue)
        gradientToOutput = -outputValue
        self._layer.backward(gradientToOutput)
        self._layer.weight.update()
        gradient = self._layer.weight.getGradient()
        print gradient
        gradientNumerical = np.zeros(gradient.size)
        weight = self._layer.weight.get()
        weightReshape = weight.reshape(weight.size)
        for i in range(weightReshape.size):
            weightReshape[i] += self._epsilon
            weightTmp = weightReshape.reshape(weight.shape)
            self._layer.weight.set(weightTmp)
            outputValueTmpPlus = self._layer.forward(inputValue)
            lossPlus = .5 * np.sum(outputValueTmpPlus ** 2)
            weightReshape[i] -= 2 * self._epsilon
            weightTmp = weightReshape.reshape(weight.shape)
            self._layer.weight.set(weightTmp)
            outputValueTmpMinus = self._layer.forward(inputValue)
            lossMinus = .5 * np.sum(outputValueTmpMinus ** 2)
            gradientNumerical[i] = (lossPlus - lossMinus)\
                                   / 2 / self._epsilon
        return gradient, gradientNumerical.reshape(gradient.shape)

    def checkGradient(self, testClass, gradient, gradientNumerical):
        print gradient / gradientNumerical
        gradientNumerical = gradientNumerical.reshape(gradient.size)
        gradient = gradient.reshape(gradient.size)
        for i in range(gradient.size):
            testClass.assertTrue(
                (gradient[i] == 0 and gradientNumerical[i] == 0) or
                (np.abs(gradient[i] / gradientNumerical[i] - 1) <
                 self._tolerance))

    def runInput(self, testClass, inputValue):
        grd, grdNum = self.computeGradientToInput(inputValue)
        self.checkGradient(testClass, grd, grdNum)

    def runWeight(self, testClass, inputValue):
        grd, grdNum = self.computeGradientToWeight(inputValue)
        self.checkGradient(testClass, grd, grdNum)

    def runAll(self, testClass, inputValue):
        self.runInput(testClass, inputValue)
        self.runWeight(testClass, inputValue)