import unittest
from concatenation_layer import ConcatenationLayer
from gradient_checker import GradientChecker
from environment import *

class ConcatenationLayerTest(unittest.TestCase):
    def test(self):
        layer = ConcatenationLayer(name='concat',
                                   numNode=0,
                                   axis=1)
        gradientChecker = GradientChecker(layer=layer)
        X = [np.random.rand(2, 2, 3), np.random.rand(2, 3, 3)]
        if USE_GPU:
            X[0] = gnp.as_garray(X[0])
            X[1] = gnp.as_garray(X[1])
        gradientChecker.runInput(self, X)

if __name__ == '__main__':
    unittest.main()
