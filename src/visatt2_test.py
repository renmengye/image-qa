import nn
import numpy as np
import unittest
import nn.stage_tests

class VisAttTest2(nn.stage_tests.StageTests):
    def setUp(self):
        self.model = nn.load('../models/visatt_2lstm.test.model.yml')
        self.eps = 1e-3
        self.tolerance = 1e-3
        # wordlstm.I
        # self.eps = 1e-1
        # self.tolerance = 1e-2
        self.stage = self.model.stageDict['attModel'].stageDict['wordlstm'].stageDict['wordlstm.I'].getStage(time=0)
        # wordlstm.F
        #self.stage = self.model.stageDict['attModel'].stageDict['wordlstm'].stageDict['wordlstm.F'].getStage(time=0)
        # wordlstm.Z
        # self.eps = 1e-3
        # self.tolerance = 1e-2
        #self.stage = self.model.stageDict['attModel'].stageDict['wordlstm'].stageDict['wordlstm.Z'].getStage(time=0)
        # wordlstm.O
        # self.eps = 1e-1
        # self.tolerance = 1e-2
        #self.stage = self.model.stageDict['attModel'].stageDict['wordlstm'].stageDict['wordlstm.O'].getStage(time=0)
        # imglstm.I
        #self.stage = self.model.stageDict['attModel'].stageDict['imglstm'].stageDict['imglstm.I'].getStage(time=0)
        # imglstm.F
        #self.stage = self.model.stageDict['attModel'].stageDict['imglstm'].stageDict['imglstm.F'].getStage(time=0)
        # imglstm.Z
        #self.stage = self.model.stageDict['attModel'].stageDict['imglstm'].stageDict['imglstm.Z'].getStage(time=0)
        # imglstm.O
        #self.stage = self.model.stageDict['attModel'].stageDict['imglstm'].stageDict['imglstm.O'].getStage(time=0)
        # attOut
        # self.eps = 1e-3
        # self.tolerance = 1e-3
        # self.stage = self.model.stageDict['attModel'].stageDict['attOut'].getStage(time=0)
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
        print dEdW
        self.chkgrd(dEdW, dEdWTmp, self.tolerance)

if __name__ == '__main__':
    unittest.main()
