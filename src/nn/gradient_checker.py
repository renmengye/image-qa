from environment import *

class GradientChecker():
    """
    Utility for checking gradient computation in a layer using finite
    difference.
    """
    def __init__(self, layer, tolerance=None, epsilon=None):
        """
        Initialize the gradient checker with configurations.
        :param layer: a subclass instance of Layer
        :param tolerance: double, relative difference allowed between
        expected and actual gradient, recommend 0.001 for CPU and 0.1 for GPU.
        :param epsilon: double, finite difference step, recommend 0.000001 for
        CPU and 0.01 for GPU.
        :return:
        """
        if tolerance is None:
            if USE_GPU:
                self._tolerance = 1e-1
            else:
                self._tolerance = 1e-3
        else:
            self._tolerance = tolerance
        if epsilon is None:
            if USE_GPU:
                self._epsilon = 1e-2
            else:
                self._epsilon = 1e-6
        else:
            self._epsilon = epsilon
        self._layer = layer

    def runGradientToInput(self, inputValue):
        outputValue = self._layer.forward(inputValue)
        gradientToOutput = outputValue
        gradient = self._layer.backward(gradientToOutput)
        return gradient

    def runGradientToWeight(self, inputValue):
        outputValue = self._layer.forward(inputValue)
        gradientToOutput = outputValue
        self._layer.backward(gradientToOutput)
        if self._layer.weight.shared:
            self._layer.weight.update()
        gradient = self._layer.weight.getGradient()
        return gradient

    def computeGradientToInput(self, inputValue, iterValue):
        """
        Compute the gradient and numerical gradient w.r.t. the input
        vector evaluated at certain input value. Internally it uses a sum of
        squares cost function with target to be the zero vector.
        :param inputValue: numpy.ndarray or gnumpy.garray or list, input value
        to the layer.
        :param iterValue: numpy.ndarray or gnumpy.garray, input value to
        iterate.
        :return: 2-tuple, gradient computed by the layer, and numerical
        gradient computed by finite difference, both in numpy.ndarray format.
        """
        iterValueReshape = iterValue.reshape(iterValue.size)
        if self._layer.gpuEnabled:
            gradientNumerical = gnp.zeros(iterValue.size)
        else:
            gradientNumerical = np.zeros(iterValue.size)
        for i in range(iterValueReshape.size):
            iterValueReshape[i] += self._epsilon
            inputValueTmp = iterValueReshape.reshape(iterValue.shape)
            outputValueTmpPlus = self._layer.forward(inputValueTmp)
            iterValueReshape[i] -= 2 * self._epsilon
            inputValueTmp = iterValueReshape.reshape(iterValue.shape)
            outputValueTmpMinus = self._layer.forward(inputValueTmp)
            if self._layer.gpuEnabled:
                lossPlus = .5 * gnp.sum(outputValueTmpPlus ** 2)
                lossMinus = .5 * gnp.sum(outputValueTmpMinus ** 2)
            else:
                lossPlus = .5 * np.sum(outputValueTmpPlus ** 2)
                lossMinus = .5 * np.sum(outputValueTmpMinus ** 2)
            gradientNumerical[i] = (lossPlus - lossMinus)\
                                   / 2 / self._epsilon
        return gradientNumerical.reshape(iterValue.shape)

    def computeGradientToWeight(self, inputValue):
        """
        Compute the gradient and numerical gradient w.r.t. the weight matrix
        evaluated at certain input value. Internally it uses a sum of squares
        cost function with target to be the zero vector.
        :param inputValue: numpy.ndarray or gnumpy.garray object, input value
        to the layer.
        :return: 2-tuple, gradient compute by the layer, and numerical
        gradient computed by finite difference, both in numpy.ndarray format.
        """
        weight = self._layer.weight.get()
        if self._layer.gpuEnabled:
            gradientNumerical = gnp.zeros(weight.size)
        else:
            gradientNumerical = np.zeros(weight.size)
        weightReshape = weight.reshape(weight.size)
        for i in range(weightReshape.size):
            weightReshape[i] += self._epsilon
            weightTmp = weightReshape.reshape(weight.shape)
            self._layer.weight.set(weightTmp)
            outputValueTmpPlus = self._layer.forward(inputValue)
            weightReshape[i] -= 2 * self._epsilon
            weightTmp = weightReshape.reshape(weight.shape)
            self._layer.weight.set(weightTmp)
            outputValueTmpMinus = self._layer.forward(inputValue)
            if self._layer.gpuEnabled:
                lossPlus = .5 * gnp.sum(outputValueTmpPlus ** 2)
                lossMinus = .5 * gnp.sum(outputValueTmpMinus ** 2)
            else:
                lossPlus = .5 * np.sum(outputValueTmpPlus ** 2)
                lossMinus = .5 * np.sum(outputValueTmpMinus ** 2)
            gradientNumerical[i] = (lossPlus - lossMinus)\
                                   / 2 / self._epsilon
        return gradientNumerical.reshape(weight.shape)

    def checkGradient(self, testClass, gradient, gradientNumerical):
        """
        Check if the gradient and numerical gradient are similar enough.
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param gradient: numpy.ndarray, gradient computed by the layer
        :param gradientNumerical: numpy.ndarray, gradient compute by finite
        difference.
        :return:
        """
        print gradient / gradientNumerical
        gradientNumerical = gradientNumerical.reshape(gradient.size)
        gradient = gradient.reshape(gradient.size)
        for i in range(gradient.size):
            testClass.assertTrue(
                (gradient[i] == 0 and gradientNumerical[i] == 0) or
                (abs(gradient[i] / gradientNumerical[i] - 1) <
                 self._tolerance))

    def runInput(self, testClass, inputValue):
        """
        Compute the gradient w.r.t. the input value evaluated at certain input
        value and check the correctness
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param inputValue: numpy.ndarray or gnumpy.garray or list, input value
        to the layer.
        :return:
        """
        grd = self.runGradientToInput(inputValue)
        if type(inputValue) is list:
            grdNum = []
            for i, inputValueItem in enumerate(inputValue):
                grdNum.append(
                    self.computeGradientToInput(inputValue, inputValueItem))
                self.checkGradient(testClass, grd[i], grdNum[i])
        else:
            grdNum = self.computeGradientToInput(inputValue, inputValue)
            self.checkGradient(testClass, grd, grdNum)

    def runWeight(self, testClass, inputValue):
        """
        Compute the gradient w.r.t. the weight matrix evaluated at certain
        input value and check the correctness.
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param inputValue: numpy.ndarray or gnumpy.garray, input value to the
        layer.
        :return:
        """
        grd = self.runGradientToWeight(inputValue)
        grdNum = self.computeGradientToWeight(inputValue)
        self.checkGradient(testClass, grd, grdNum)

    def runAll(self, testClass, inputValue):
        """
        Compute the gradient w.r.t. the weight matrix and the input value
        evaluated at certain input value and check the correctness.
        :param testClass: A unittest.TestCase subclass that has assertTrue
        method.
        :param inputValue: numpy.ndarray or gnumpy.garray, input value to the layer
        :return:
        """
        self.runInput(testClass, inputValue)
        self.runWeight(testClass, inputValue)
