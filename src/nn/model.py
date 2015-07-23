from identity_layer import *

class Model():
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
        self.loss = 0
        self._inputLayer = IdentityLayer(name='input')
        self._targetLayer = IdentityLayer(name='target')
        self._outputLayer = outputLayer
        self._lossLayer = lossLayer
        for layer in inputLayers:
            layer.addInput(self._inputLayer)

        if lossLayer is not None:
            self._lossLayer.addInput(self._outputLayer)
            self._lossLayer.addInput(self._targetLayer)
            self.layers = self._computeTraversalOrder(lossLayer)
        else:
            # Inference model only
            self.layers = self._computeTraversalOrder(outputLayer)
        # Initialize weights
        for layer in self.layers:
            layer.init()

    @staticmethod
    def _computeTraversalOrder(lastLayer):
        """
        Requires all layers have unique name.
        :param lastLayer:
        :return:
        """
        workingList = [lastLayer]
        traversedMap = {}
        counter = 0
        while len(workingList) > 0:
            currentLayer = workingList[0]
            if not traversedMap.has_key(currentLayer):
                traversedMap[currentLayer] = counter
                counter += 1
            workingList.remove(currentLayer)
            for previousLayer in currentLayer.inputLayers:
                # Fix if the children is ahead of the parent.
                if traversedMap.has_key(previousLayer):
                    traversedMap.pop(previousLayer)
                workingList.append(previousLayer)
        sortedOrder = sorted(traversedMap.keys(),
                             key=lambda x:-traversedMap[x])
        return sortedOrder


    def trainStep(self, inputValue, targetValue, weights=None):
        """

        :param inputValue:
        :param targetValue: 
        :param weights: numpy.ndarray or gnumpy.garray, weights of each
        examples.
        :return:
        """
        self._inputLayer.setValue(inputValue)
        self._targetLayer.setValue(targetValue)
        for layer in self.layers:
            layer.graphForward()
        self.loss = self._lossLayer.loss
        for layer in reversed(self.layers):
            layer.graphBackward()

    def runOnce(self, inputValue):
        """
        Run inference in the network.
        :param inputValue: Input values
        :return:
        """
        self._inputLayer.setValue(inputValue)
        # Inference
        for layer in self.layers:
            if layer is not self._lossLayer:
                layer.graphForward()
        return self._outputLayer.getValue()

    def toDict(self, filename):
        pass

    @staticmethod
    def fromDict(self, filename):
        pass

    def serializeWeights(self):
        pass

    def loadWeights(self, weights):
        pass