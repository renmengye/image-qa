from container import *

class Sequential(Container):
    def forward(self, X, dropout=True):
        X1 = X
        for stage in self.stages:
            if hasattr(stage, 'dropout'):
                stage.dropout = dropout
                X1 = stage.forward(X1)
            else:
                X1 = stage.forward(X1)
        return X1

    def backward(self, dEdY):
        for stage in reversed(self.stages):
            dEdY = stage.backward(dEdY)
            if dEdY is None: break
        return dEdY if self.outputdEdX else None