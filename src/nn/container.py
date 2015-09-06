from identity_layer import IdentityLayer
from layer import Layer, WeightLayer

###
# Let's support multiple output layers. Just do multiple traversals.
###
class Container(Layer):
    def __init__(self,
                 name,
                 inputLayers,
                 outputLayer):
        """
        Initialize a model
        :param name: Name of the layer.
        :param inputLayers: a list of layers, which are subclasses of Layer,
        input nodes of the model.
        :param outputLayer: a list of layers, which are subclasses of Layer,
        output nodes of the model. The output is only one layer.
        :return:
        """
        Layer.__init__(name=name,
                       numNode=outputLayer.numNode)
        self._inputLayer = IdentityLayer(name='input',
                                         numNode=inputLayers[0].numNode)
        self._outputLayer = outputLayer
        self._weights = {}
        for layer in inputLayers:
            layer.addInput(self._inputLayer)

        self.layers = self._computeTraversalOrder(outputLayer)

        # Check for layer name conflict
        self.layerDict = {}
        for layer in self.layers:
            if layer.name in self.layerDict:
                raise Exception('Layer name conflict: ' + layer.name)
            else:
                layer.name = self.name + ':' + layer.name
                self.layerDict[layer.name] = layer

    def initialize(self):
        """
        Initialize weight.
        :return:
        """
        for layer in self.layers:
            layer.initialize()

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

    def graphForward(self):
        self._inputValue = self.getInput()
        self._outputValue = self.forward(self._inputValue)

    def forward(self, inputValue):
        pass

    def backward(self, gradientToOutput):
        pass

    def serializeWeight(self):
        allWeights = {}
        for layer in self.layers:
            if type(layer) is WeightLayer:
                layerWeights = layer.serializeWeight()
                for key in layerWeights.iterkeys():
                    if key in allWeights:
                        if not (allWeights[key].shared and
                                layerWeights[key].shared):
                            raise Exception('Non-shared weights name conflict.')
                    else:
                        allWeights[key] = layerWeights[key]
        return allWeights

    def deserializeWeight(self, weight):
        """

        :param weight: Dictionary with keys to be weight name and value to be
        weight value.
        :return:
        """
        for layer in self.layers:
            layer.deserializeWeights(weight)

    def toDict(self, filename):
        pass

    @staticmethod
    def fromDict(self, filename):
        pass
