import nn
import yaml
import numpy as np
from train import readFlags

'''
Usage: python xvalid.py {name} -d {train data} -m {model spec} -c {config} -o {output folder}
'''

if __name__ == '__main__':
    name, modelFilename, configFilename, trainDataFilename, outputFolder = readFlags()
    trainData = np.load(trainDataFilename)
    trainInput = trainData[0]
    trainTarget = trainData[1]
    trainOpt = yaml.load(configFilename)

    for i in range(0, 10):
        trainOpt['xvalidNo'] = i
        model = nn.load(modelFilename)
        trainer = nn.Trainer(
            name=name + ('-%d' % i),
            model=model,
            trainOpt=trainOpt,
            outputFolder=outputFolder
        )
        trainer.train(trainInput, trainTarget)