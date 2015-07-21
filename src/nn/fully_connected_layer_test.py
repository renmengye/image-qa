from fully_connected_layer import *
from activation_fn import *
from weight import *
from weight_initializer import *
from gradient_checker import *
import unittest

class FullyConnectedLayerTest(unittest.TestCase):
    def test(self):
        layer = FullyConnectedLayer(name='fc',
                                    activationFn=SigmoidActivationFn(),
                                    weight=Weight(
                                        initializer=UniformWeightInitializer(
                                            limit=[-0.5, 0.5],
                                            seed=2,
                                            shape=[5, 6],
                                            affine=True),
                                        gdController=None,
                                        shared=False))

        gradientChecker = GradientChecker(layer=layer)
        X = np.random.rand(5, 4)
        if USE_GPU:
            X = gnp.as_garray(X)
        gradientChecker.runAll(self, X)

if __name__ == '__main__':
    unittest.main()