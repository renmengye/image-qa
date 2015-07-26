from elementwise_layer import ElementwiseProductLayer, ElementwiseSumLayer
from gradient_checker import GradientChecker
import unittest
from environment import *

class ElementwiseLayerTest(unittest.TestCase):
    def testProduct(self):
        layer = ElementwiseProductLayer(name='product', numNode=0)
        gradientChecker = GradientChecker(layer=layer)
        X = [np.random.rand(5, 4), np.random.rand(5, 4)]
        if USE_GPU:
            X[0] = gnp.as_garray(X[0])
            X[1] = gnp.as_garray(X[1])
        gradientChecker.runInput(self, X)

    def testSum(self):
        layer = ElementwiseSumLayer(name='sum', numNode=0)
        gradientChecker = GradientChecker(layer=layer)
        X = [np.random.rand(5, 4), np.random.rand(5, 4)]
        if USE_GPU:
            X[0] = gnp.as_garray(X[0])
            X[1] = gnp.as_garray(X[1])
        gradientChecker.runInput(self, X)

if __name__ == '__main__':
    unittest.main()
