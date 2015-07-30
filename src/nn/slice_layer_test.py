import unittest
from slice_layer import SliceLayer
from gradient_checker import GradientChecker
from environment import *

class SliceLayerTest(unittest.TestCase):
    def test(self):
        layer = SliceLayer(name='slice',
                           numNode=0,
                           start=1,
                           end=3,
                           axis=2)
        gradientChecker = GradientChecker(layer=layer)
        X = np.random.rand(2, 2, 4)
        if USE_GPU:
            X = gnp.as_garray(X)
        gradientChecker.runInput(self, X)

if __name__ == '__main__':
    unittest.main()
