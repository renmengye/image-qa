import yaml
import sys
import numpy as np
from trainer import *

'''
Usage: python xvalid.py {name} {config} {train data} {output folder}
'''

name = sys.argv[1]
configFilename = sys.argv[2]
trainDataFilename = sys.argv[3]
outputFolder = sys.argv[4]

trainData = np.load(trainDataFilename)
trainInput = trainData[0]
trainTarget = trainData[1]

with open(configFilename) as f:
	configTxt = f.read()

for i in range(0, 10):
	if i > 0:
		configTxt = configTxt.replace('xvalidNo: %d' % (i-1), 'xvalidNo: %d' % i)
	tmpConfigFilename = configFilename + '.tmp'
	with open(tmpConfigFilename, 'w+') as f:
		f.write(configTxt)

	pipeline = Trainer.initFromConfig(
	    name=name + ('-%d' % i),
	    configFilename=tmpConfigFilename,
	    outputFolder=outputFolder
	)

	pipeline.train(trainInput, trainTarget)