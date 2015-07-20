import os
import numpy as np

class VocabDict:
    """
    A vocabulary dictionary implementation.
    Storage is build by a list of strings.
    Can be used to encode text into word IDs and decode word IDs into text.
    """
    def __init__(self,
                 filename,
                 mode='r',
                 offset=1,
                 unknownToken='<UNK>'):
        """
        Initialize a vocabulary dictionary
        :param filename: File name of the dictionary, prefer ".npy"
        :param mode: Mode 'r' or 'w', by default 'r' which will load the dataset.
        :param offset: Word index offset (or start ID).
        :return:
        """
        self.filename = filename
        self.offset = offset
        if os.path.exists(filename) and mode == 'r':
            self.idict = np.load(filename)
            self.dict = self._invertIdict(self.idict, offset=offset)
        else:
            self.unknownToken = unknownToken
            self.idict = []
            self.dict = {}

    def buildDict(self, text):
        """
        Build a vocabulary dictionary
        :param text: List of strings (sentences).
        :return:
        """
        self.idict.append(self.unknownToken)
        self.dict = {}
        self.idict = []
        for sentence in text:
            for word in sentence.split(' '):
                if not self.dict.has_key(word):
                    self.dict[word] = len(self.idict) + self.offset
                    self.idict.append(word)
        self.dict[self.unknownToken] = len(self.idict) + self.offset
        self.idict.append(self.unknownToken)

    @staticmethod
    def _invertIdict(idict, offset=1):
        """
        Invert the dictionary to get a map from word to ID.
        :param idict: A list which index maps to a word
        :param offset: ID which the first element in the list maps to
        :return:
        """
        dict = {}
        for i, word in enumerate(idict):
            dict[word] = offset + i
        return dict

    def encode(self, text, maxlen=0):
        """
        Encode a list of sentence into a list of IDs.
        :param text: A list of strings.
        :param maxlen: Maximum length, if 0 will compute the maximum length of
        the resulting
        :return: A 2D NxM numpy array with N to be number of sentences,
        M to be the maximum length
        """
        if maxlen == 0:
            # Find the maximum length of the text.
            maxlen = self._findMaxlen(text)
            pass

        result = np.zeros((len(text), maxlen), dtype='int')
        for nsent, sentence in enumerate(text):
            # sentenceIds = []
            for nword, word in enumerate(sentence.split(' ')):
                if self.dict.has_key(word):
                    result[nsent, nword] = self.dict[word]
                else:
                    result[nsent, nword] = self.dict[self.unknownToken]
        return result

    @staticmethod
    def _findMaxlen(text):
        maxlen = 0
        for sentence in text:
            length = len(sentence.split(' '))
            if length > maxlen: maxlen = length
        return maxlen

    def decode(self, ids):
        """
        Decode word IDs back to sentence form.
        :param ids: 2D NxM numpy array.
        :return: A list of strings in sentence form.
        """
        text = []
        for n in ids.shape[0]:
            words = []
            for i in ids.shape[1]:
                words.append(self.idict[ids[n, i] + self.offset])
            text.append(' '.join(words))
        return text

    def save(self):
        np.save(self.filename, self.idict)