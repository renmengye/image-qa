import nn
import nn.valid_tool as vt
import yaml
import numpy as np
from train import readFlags

'''
Usage: python xvalid.py {name} -d {train data} -m {model spec} -c {config} -o {output folder}
'''

if __name__ == '__main__':
    name, modelFilename, configFilename, trainDataFilename, testDataFilename, outputFolder = readFlags()
    trainData = np.load(trainDataFilename)
    trainInput = trainData[0]
    trainTarget = trainData[1]
    trainInput, trainTarget = vt.shuffleData(
        trainInput, trainTarget, np.random.RandomState(2))
    with open(configFilename) as f:
        trainOpt = yaml.load(f)

    for i in range(0, 10):
        trainInput_, trainTarget_, testInput_, testTarget_ = \
        vt.splitData(trainInput, trainTarget, 0.1, i)
        trainOpt['heldOutRatio'] = 0.1
        trainOpt['xvalidNo'] = 0
        trainOpt['needValid'] = True

        model = nn.load(modelFilename)
        trainer = nn.Trainer(
            name=name + ('-%d-v' % i),
            model=model,
            trainOpt=trainOpt,
            outputFolder=outputFolder
        )
        trainer.train(trainInput_, trainTarget_)

        # Train again with all data, without validation
        trainOpt['needValid'] = False
        trainOpt['numEpoch'] = trainer.stoppedEpoch + 1
        trainer = nn.Trainer(
            name=name + ('-%d' % i),
            model=model,
            trainOpt=trainOpt,
            outputFolder=outputFolder
        )
        trainer.train(trainInput_, trainTarget_)
        testOutput = nn.test(model, testInput_)
        testRate, correct, total = nn.calcRate(model, testOutput, testTarget_)
        with open(os.path.join(trainer.outputFolder, 'result.txt'), 'w+') as f:
            f.write('Test rate: %f' % testRate)
