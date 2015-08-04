from embedding_layer import EmbeddingLayer
from weight import Weight
from weight_initializer import UniformWeightInitializer
from gradient_checker import GradientChecker
from environment import *
import unittest

class EmbeddingLayerTest(unittest.TestCase):
    def test(self):
        layer = EmbeddingLayer(name='emb',
                               inputDim=3,
                               numNode=4,
                               weight=Weight(
                                   name='emb',
                                   initializer=UniformWeightInitializer(
                                       limit=[-0.5, 0.5],
                                       seed=2),
                                   controller=None,
                                   shared=False))
        layer.initialize()
        gradientChecker = GradientChecker(layer=layer)
        X = (np.random.rand(5, 1) * 3).astype('int') + 1
        gradientChecker.runWeight(self, X)

if __name__ == '__main__':
    unittest.main()
