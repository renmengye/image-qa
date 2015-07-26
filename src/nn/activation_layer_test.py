from activation_layer import ActivationLayer
from activation_fn import SigmoidActivationFn
from gradient_checker import GradientChecker
import unittest
from environment import *

class ActivationLayerTest(unittest.TestCase):
    def test(self):
        layer = ActivationLayer(name='act',
                                numNode=0,
                                activationFn=SigmoidActivationFn())
        gradientChecker = GradientChecker(layer=layer)
        X = np.random.rand(5, 4)
        if USE_GPU:
            X = gnp.as_garray(X)
        gradientChecker.runInput(self, X)

if __name__ == '__main__':
    unittest.main()
