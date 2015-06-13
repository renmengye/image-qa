import numpy as np
import os

class Dataset:
    """
    Dataset object for storing everything you need.
    Not optimized for large arrays or sparse arrays.
    Consider storing the ID instead.
    """
    def __init__(self, filename, mode='r'):
        """
        Initialize a dataset object
        :param filename: File name of the dataset,
        including extension, preferably ".npz".
        :param mode: Mode 'r' or 'w',
        by default 'r' which will load the dataset.
        :return:
        """
        self.data = {}
        self.mode = mode
        self.filename = filename
        if os.path.exists(filename) and mode == 'r':
            self.data = np.load(filename)

    def get(self, key):
        return self.data[key]

    def getTrainInput(self):
        return self.get('trainInput')

    def getTrainTarget(self):
        return self.get('trainTarget')

    def getTrainInputWeights(self):
        return self.get('trainInputWeights')

    def getValidInput(self):
        return self.get('validInput')

    def getValidTarget(self):
        return self.get('validTarget')

    def getValidInputWeights(self):
        return self.get('validInputWeights')

    def getTestInput(self):
        return self.get('testInput')

    def getTestTarget(self):
        return self.get('testTarget')

    def getTestInputWeights(self):
        return self.get('testInputWeights')

    def set(self, key, obj):
        self.data[key] = obj

    def setTrainInput(self, obj):
        self.set('trainInput', obj)

    def setTrainTarget(self, obj):
        self.set('trainTarget', obj)

    def setValidInput(self, obj):
        self.set('validInput', obj)

    def setValidTarget(self, obj):
        self.set('validTarget', obj)

    def setTestInput(self, obj):
        self.set('testInput', obj)

    def setTestTarget(self, obj):
        self.set('testTarget', obj)

    def contains(self, key):
        return key in self.data

    def containsTrain(self):
        return self.contains('trainInput')

    def containsValid(self):
        return self.contains('validInput')

    def containsTest(self):
        return self.contains('testInput')

    def containsInputWeights(self):
        return self.contains('trainInputWeights')

    def save(self):
        np.savez(self.filename, **self.data)