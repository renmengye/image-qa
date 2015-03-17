import nn
import numpy as np
import unittest
import nn.stage_tests

class ImgWordTest(nn.stage_tests.StageTests):
    def setUp(self):
        self.eps = 1e-3
        self.tolerance = 1e-2
        self.model = nn.load('../models/imgword.test.model.yml')
        self.model.stageDict['dropout'].debug = True
        # lstm.I
        #self.stage = self.model.stageDict['lstm'].stageDict['lstm.I'].getStage(time=0)
        # lstm.F
        #self.stage = self.model.stageDict['lstm'].stageDict['lstm.F'].getStage(time=0)
        # lstm.Z
        #self.stage = self.model.stageDict['lstm'].stageDict['lstm.Z'].getStage(time=0)
        # lstm.O
        #self.stage = self.model.stageDict['lstm'].stageDict['lstm.O'].getStage(time=0)
        # Answer softmax
        #self.stage = self.model.stageDict['softmax']
        print self.stage.name
        self.testInputErr = False
        self.costFn = nn.crossEntIdx

    def test_grad(self):
        data = np.load('../data/imgword/train-37-unk.npy')
        X = data[0][0:10]
        T = data[1][0:10]
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T, self.eps)
        print dEdW/dEdWTmp
        # print dEdW
        # print dEdWTmp
        self.chkgrd(dEdW, dEdWTmp, self.tolerance)

if __name__ == '__main__':
    unittest.main()