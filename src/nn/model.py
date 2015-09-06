from layer import WeightLayer
from container import Container
from identity_layer import IdentityLayer

class Model(Container):
    """
    bunch of layers.
    output value.
    determine the order of traversal.
    serialize.
    load.
    """
    def __init__(self, inputLayers, outputLayer, lossLayer=None):
        """
        Initialize a model
        :param inputLayers: a list of layers, which are subclasses of Layer,
        input nodes of the model.
        :param outputLayer: a list of layers, which are subclasses of Layer,
        output nodes of the model. The output is only one layer.
        :param lossLayer: subclass of Layer, loss layer that takes in target,
        the last layer of the model.
        :return:
        """
        self._inputLayer = IdentityLayer(name='input',
                                         numNode=inputLayers[0].numNode)
        self._targetLayer = IdentityLayer(name='target',
                                          numNode=outputLayer.numNode)
        self._outputLayer = outputLayer
        self._lossLayer = lossLayer
        self._weights = {}
        for layer in inputLayers:
            layer.addInput(self._inputLayer)

        if lossLayer is not None:
            self._lossLayer.addInput(self._outputLayer)
            self._lossLayer.addInput(self._targetLayer)
            self.layers = self._computeTraversalOrder(lossLayer)
        else:
            # Inference model only
            self.layers = self._computeTraversalOrder(outputLayer)

        # Check for layer name conflict
        self.layerDict = {}
        for layer in self.layers:
            if layer.name in self.layerDict:
                raise Exception('Layer name conflict: ' + layer.name)
            else:
                self.layerDict[layer.name] = layer

    def trainStep(self, inputValue, targetValue, exampleWeights=None):
        """

        :param inputValue:
        :param targetValue: 
        :param exampleWeights: numpy.ndarray or gnumpy.garray, weights of each
        examples.
        :return:
        """
        self._inputLayer.setValue(inputValue)
        self._targetLayer.setValue(targetValue)
        for layer in self.layers:
            layer.graphForward()
        for layer in reversed(self.layers):
            layer.graphBackward()

    def forward(self, inputValue):
        """
        Forward pass.
        :param inputValue: Input values
        :return:
        """
        self._inputLayer.setValue(inputValue)
        # Inference
        for layer in self.layers:
            if layer is not self._lossLayer:
                layer.graphForward()
        return self._outputLayer.getValue()

    def backward(self):
        """

        :return:
        """
        pass

    def getLoss(self):
        return self._lossLayer.getLoss()

