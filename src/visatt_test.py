import nn
import numpy as np
import unittest
import nn.stage_tests

class VisAttTest(nn.stage_tests.StageTests):
    def setUp(self):
        self.eps = 1e-3
        self.tolerance = 1e-3
        self.model = nn.load('../models/visatt_mlp.test.model.yml')
        # txtAttCtx
        #self.stage = self.model.stageDict['attModel'].stageDict['txtAttCtx'].getStage(time=0)
        # lstm.I
        #self.stage = self.model.stageDict['attModel'].stageDict['lstm'].stageDict['lstm.I'].getStage(time=0)
        # lstm.F
        #self.stage = self.model.stageDict['attModel'].stageDict['lstm'].stageDict['lstm.F'].getStage(time=0)
        # lstm.Z
        #self.stage = self.model.stageDict['attModel'].stageDict['lstm'].stageDict['lstm.Z'].getStage(time=0)
        # lstm.O
        #self.eps = 1e-3
        #self.tolerance = 1e-3
        #self.stage = self.model.stageDict['attModel'].stageDict['lstm'].stageDict['lstm.O'].getStage(time=0)
        # attHid1
        #self.eps = 0.1
        #self.tolerance = 1e-2
        #self.stage = self.model.stageDict['attModel'].stageDict['attHid1'].getStage(time=0)
        # attOut
        #self.eps = 1e-3
        #self.tolerance = 1e-3
        #self.stage = self.model.stageDict['attModel'].stageDict['attOut'].getStage(time=0)
        # Answer softmax
        #self.stage = self.model.stageDict['answer']
        print self.stage.name
        self.testInputErr = False
        self.costFn = nn.crossEntIdx

    def test_grad(self):
        data = np.load('../data/imgword/train-37-unk-att.npy')
        X = data[0][0:10]
        T = data[1][0:10]
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T, self.eps)
        print dEdW/dEdWTmp
        self.chkgrd(dEdW, dEdWTmp, self.tolerance)

if __name__ == '__main__':
    unittest.main()