import sys
from pipeline import *

name = sys.argv[1]
outputFolder = None
for i in range(2, len(sys.argv)):
    if sys.argv[i] == '-o':
        outputFolder = sys.argv[i + 1]
    elif sys.argv[i] == '-d':
        trainDataFilename = sys.argv[i + 1]
    elif sys.argv[i] == '-c':
        configFilename = sys.argv[i + 1]

trainData = np.load(trainDataFilename)
trainInput = trainData[0]
trainTarget = trainData[1]

pipeline = Pipeline.initFromConfig(
    name=name,
    configFilename=configFilename,
    outputFolder=outputFolder
)

pipeline.train(trainInput, trainTarget)