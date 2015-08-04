from model import Model
from layer import Layer
from fully_connected_layer import FullyConnectedLayer
from weight import Weight
from weight_initializer import UniformWeightInitializer
from mse_loss_layer import MSELossLayer
from activation_fn import SigmoidActivationFn
from environment import *
import unittest


class ModelTest(unittest.TestCase):
    def testTraverseSequential(self):
        lastLayer = Layer(name='layer1',
                          numNode=0).connect(
            Layer(name='layer2',
                  numNode=0)).connect(
            Layer(name='layer3',
                  numNode=0))
        order = Model._computeTraversalOrder(lastLayer)
        self.assertEqual(3, len(order))
        self.assertEqual('layer1', order[0].name)
        self.assertEqual('layer2', order[1].name)
        self.assertEqual('layer3', order[2].name)

    def testTraverseParallel(self):
        layer1 = Layer(name='layer1', numNode=0)
        layer2a = layer1.connect(
            Layer(name='layer2a',
                  numNode=0))
        layer2b = layer1.connect(
            Layer(name='layer2b',
                  numNode=0))
        layer3 = layer2a.connect(
            Layer(name='layer3',
                  numNode=0))
        layer3.addInput(layer2b)
        order = Model._computeTraversalOrder(layer3)
        self.assertEqual(4, len(order))
        self.assertEqual('layer1', order[0].name)
        self.assertEqual('layer3', order[3].name)

    def testTraverseComplex(self):
        layer1 = Layer(name='layer1', numNode=0)
        layer2 = layer1.connect(Layer(name='layer2', numNode=0))
        layer4 = layer2\
            .connect(Layer(name='layer3', numNode=0))\
            .connect(Layer(name='layer4', numNode=0))
        layer4.addInput(layer2)
        layer5 = layer2.connect(Layer(name='layer5', numNode=0))
        layer6 = layer5.connect(Layer(name='layer6', numNode=0))
        layer6.addInput(layer4)
        order = Model._computeTraversalOrder(layer6)
        self.assertEqual(6, len(order))
        self.assertEqual('layer1', order[0].name)
        self.assertEqual('layer2', order[1].name)
        self.assertTrue(order[2].name == 'layer3' or order[2].name == 'layer5')
        if order[2].name == 'layer3':
            self.assertEqual('layer4', order[3].name)
            self.assertEqual('layer5', order[4].name)
        elif order[2].name == 'layer5':
            self.assertEqual('layer3', order[3].name)
            self.assertEqual('layer4', order[4].name)
        self.assertEqual('layer6', order[5].name)

    def testMLP(self):
        inputLayer = FullyConnectedLayer(name='fc1',
                                    activationFn=SigmoidActivationFn(),
                                    numNode=4,
                                    weight=Weight(
                                        name='fc1',
                                        initializer=UniformWeightInitializer(
                                            limit=[-0.5, 0.5],
                                            seed=2),
                                        controller=None,
                                        shared=False))
        outputLayer = inputLayer.connect(
            FullyConnectedLayer(name='fc2',
                                activationFn=SigmoidActivationFn(),
                                numNode=3,
                                weight=Weight(name='fc2',
                                    initializer=UniformWeightInitializer(
                                    limit=[-0.5, 0.5],
                                            seed=2),
                                        controller=None,
                                        shared=False)))\
                                .connect(FullyConnectedLayer(name='fc3',
                                     activationFn=SigmoidActivationFn(),
                                     numNode=5,
                                     weight=Weight(
                                         name='fc2',
                                        initializer=UniformWeightInitializer(
                                            limit=[-0.5, 0.5],
                                            seed=2),
                                        controller=None,
                                        shared=False)))
        model = Model(inputLayers=[inputLayer], outputLayer=outputLayer,
                      lossLayer=MSELossLayer(name='mse'))
        X = np.random.rand(5, 4)
        T = np.zeros((5, 5))
        if USE_GPU:
            X = gnp.as_garray(X)
        Y = model.runOnce(X)
        self.assertEqual((5, 5), Y.shape)
        dLdX = model.trainStep(X, T)
        self.assertGreater(model.getLoss(), 0)

if __name__ == '__main__':
    unittest.main()
