from container import *

class Sequential(Container):
    def backward(self, dEdY):
        for stage in reversed(self.stages):
            dEdY = stage.backward(dEdY)
            if dEdY is None: break
        return dEdY if self.outputdEdX else None