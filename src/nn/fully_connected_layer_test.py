from fully_connected_layer import FullyConnectedLayer
from activation_fn import SigmoidActivationFn
from weight import Weight
from weight_initializer import UniformWeightInitializer
from gradient_checker import GradientChecker
from environment import *
import unittest

class FullyConnectedLayerTest(unittest.TestCase):
    def test(self):
        layer = FullyConnectedLayer(name='fc',
                                    activationFn=SigmoidActivationFn(),
                                    numNode=4,
                                    weight=Weight(
                                        name='fc',
                                        initializer=UniformWeightInitializer(
                                            limit=[-0.5, 0.5],
                                            seed=2),
                                        controller=None,
                                        shared=False))
        layer.initialize(inputNumNode=4)
        gradientChecker = GradientChecker(layer=layer)
        X = np.random.rand(5, 4)
        if USE_GPU:
            X = gnp.as_garray(X)
        gradientChecker.runAll(self, X)

if __name__ == '__main__':
    unittest.main()
