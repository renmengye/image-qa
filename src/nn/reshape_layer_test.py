import unittest
from reshape_layer import ReshapeLayer
from gradient_checker import GradientChecker
from environment import *

class ConcatenationLayerTest(unittest.TestCase):
    def test(self):
        layer = ReshapeLayer(name='reshape',
                             numNode=0,
                             reshapeFn='(x[0], x[1] * x[2])')
        gradientChecker = GradientChecker(layer=layer)
        X = np.random.rand(2, 2, 3)
        if USE_GPU:
            X = gnp.as_garray(X)
        gradientChecker.runInput(self, X)

if __name__ == '__main__':
    unittest.main()
