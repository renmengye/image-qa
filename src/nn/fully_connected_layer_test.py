from fully_connected_layer import *
from activation_fn import SigmoidActivationFn
from weight import Weight
from weight_initializer import UniformWeightInitializer
from gradient_checker import GradientChecker
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
                                        gdController=None,
                                        shared=False))
        layer.init(inputNumNode=4)
        if USE_GPU:
            epsilon = 1e-2
            tolerance = 1e-1
        else:
            epsilon = 1e-5
            tolerance = 1e-4
        gradientChecker = GradientChecker(
                            layer=layer, 
                            epsilon=epsilon, 
                            tolerance=tolerance)
        X = np.random.rand(5, 4)
        if USE_GPU:
            X = gnp.as_garray(X)
        gradientChecker.runAll(self, X)

if __name__ == '__main__':
    unittest.main()
