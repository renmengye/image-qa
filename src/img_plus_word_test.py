import nn
import numpy as np
import unittest
import nn.stage_tests

class ImgPlusWordTest(nn.stage_tests.StageTests):
    def setUp(self):
        self.eps = 1e-3
        self.tolerance = 1e-3
        self.model = nn.load('../models/img_plus_word.test.model.yml')
        #self.model = nn.load('../models/imgword_2i2w.test.model.yml')
        self.model.stageDict['dropout'].debug = True
        # LSTM
        self.stage = self.model.stageDict['lstm.I-0']
        # Answer softmax
        #self.stage = self.model.stageDict['answer']
        #self.stage = self.model.stageDict['softmax']
        print self.stage.name
        self.testInputErr = False
        self.costFn = nn.crossEntIdx

    def test_grad(self):
        #data = np.load('../data/daquar-37/train-unk.npy')
        data = np.load('../data/daquar-37/train-unk-att.npy')
        X = data[0][0:10]
        T = data[1][0:10]
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T, self.eps)
        print dEdW/dEdWTmp
        self.chkgrd(dEdW, dEdWTmp, self.tolerance)

if __name__ == '__main__':
    unittest.main()