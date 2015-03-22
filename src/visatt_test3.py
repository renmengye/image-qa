import nn
import numpy as np
import unittest
import nn.stage_tests

class VisAttTest3(nn.stage_tests.StageTests):
    def setUp(self):
        self.eps = 1e-3
        self.tolerance = 1e-3
        self.model = nn.load('../models/visatt_sep_cos.model.yml')
        # ilstm.I
        #self.stage = self.model.stageDict['attModel'].stageDict['ilstm'].stageDict['ilstm.I'].getStage(time=0)
        # ilstm.F
        #self.stage = self.model.stageDict['attModel'].stageDict['ilstm'].stageDict['ilstm.F'].getStage(time=0)
        # ilstm.Z
        #self.stage = self.model.stageDict['attModel'].stageDict['ilstm'].stageDict['ilstm.Z'].getStage(time=0)
        # ilstm.O
        #self.eps = 1e-3
        #self.tolerance = 1e-3
        #self.stage = self.model.stageDict['attModel'].stageDict['ilstm'].stageDict['ilstm.O'].getStage(time=0)

        # wlstm1.I
        #self.stage = self.model.stageDict['wlstm1'].stageDict['wlstm1.I'].getStage(time=0)
        # wlstm1.F
        #self.stage = self.model.stageDict['wlstm1'].stageDict['wlstm1.F'].getStage(time=0)
        # wlstm1.Z
        #self.stage = self.model.stageDict['wlstm1'].stageDict['wlstm1.Z'].getStage(time=0)
        # wlstm1.O
        #self.stage = self.model.stageDict['wlstm1'].stageDict['wlstm1.O'].getStage(time=0)


        # wlstm2.I
        #self.stage = self.model.stageDict['wlstm2'].stageDict['wlstm2.I'].getStage(time=0)
        # wlstm2.F
        #self.stage = self.model.stageDict['wlstm2'].stageDict['wlstm2.F'].getStage(time=0)
        # wlstm2.Z
        #self.stage = self.model.stageDict['wlstm2'].stageDict['wlstm2.Z'].getStage(time=0)
        # wlstm2.O
        #self.stage = self.model.stageDict['wlstm2'].stageDict['wlstm2.O'].getStage(time=0)

        # attHid1
        #self.eps = 0.1
        #self.tolerance = 1e-2
        #self.stage = self.model.stageDict['attModel'].stageDict['attBeta'].getStage(time=0)
        #self.stage = self.model.stageDict['attModel'].stageDict['attHid1'].getStage(time=0)
        # attOut
        #self.eps = 1e-3
        #self.tolerance = 1e-3
        #self.stage = self.model.stageDict['attModel'].stageDict['attOut'].getStage(time=0)
        # Answer softmax
        #self.stage = self.model.stageDict['answer']
        #self.stage = self.model.stageDict['outputMap']
        print self.stage.name
        self.testInputErr = False
        self.costFn = nn.rankingLoss
        #self.costFn = nn.crossEntIdx

    def test_grad(self):
        data = np.load('../data/imgword/train-37-unk-att.npy')
        X = data[0][0:10]
        T = data[1][0:10]
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T, self.eps)
        print dEdW
        #print dEdWTmp
        print dEdW/dEdWTmp
        self.chkgrd(dEdW, dEdWTmp, self.tolerance)

if __name__ == '__main__':
    unittest.main()