import nn
import numpy as np
import sys
import yaml

'''
Usage: python train.py {name} -d {train data} -m {model spec} -c {config} -o {output folder}
'''

def readFlags():
    if len(sys.argv) < 2:
        raise Exception('Name not specified')
    name = sys.argv[1]
    outputFolder = None
    configFilename = None
    trainDataFilename = None
    modelFilename = None
    for i in range(2, len(sys.argv) - 1):
        if sys.argv[i] == '-o' or sys.argv[i] == '-out':
            outputFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-d' or sys.argv[i] == '-data':
            trainDataFilename = sys.argv[i + 1]
        elif sys.argv[i] == '-m' or sys.argv[i] == '-model':
            modelFilename = sys.argv[i + 1]
        elif sys.argv[i] == '-c' or sys.argv[i] == '-config':
            configFilename = sys.argv[i + 1]

    if configFilename is None:
        raise Exception('Config file not specified')
    if trainDataFilename is None:
        raise Exception('Data file not specified')
    if modelFilename is None:
        raise Exception('Model file not specified')

    return name, modelFilename, configFilename, trainDataFilename, outputFolder

if __name__ == '__main__':
    name, modelFilename, configFilename, trainDataFilename, outputFolder = readFlags()
    trainOpt = yaml.load(configFilename)
    trainData = np.load(trainDataFilename)
    trainInput = trainData[0]
    trainTarget = trainData[1]
    model = nn.load(modelFilename)
    trainer = nn.Trainer(
        name=name,
        model=model,
        trainOpt=trainOpt,
        outputFolder=outputFolder
    )

    trainer.train(trainInput, trainTarget)