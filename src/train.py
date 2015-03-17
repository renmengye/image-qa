import nn
import numpy as np
import os
import sys
import yaml
import email

'''
Usage: python train.py {name} -d {train data} -m {model spec} -c {config} -o {output folder}
'''

def readFlags():
    params = {}
    if len(sys.argv) < 2:
        raise Exception('Name not specified')
    params['name'] = sys.argv[1]

    params['outputFolder'] = None
    params['configFilename'] = None
    params['trainDataFilename'] = None
    params['testDataFilename'] = None
    params['validDataFilename'] = None
    params['allDataFilename'] = None
    params['modelFilename'] = None
    for i in range(2, len(sys.argv) - 1):
        if sys.argv[i] == '-o' or sys.argv[i] == '-out':
            params['outputFolder'] = sys.argv[i + 1]
        elif sys.argv[i] == '-d' or sys.argv[i] == '-train':
            params['trainDataFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-t' or sys.argv[i] == '-test':
            params['testDataFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-v' or sys.argv[i] == '-valid':
            params['validDataFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-a' or sys.argv[i] == '-alldata':
            params['allDataFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-m' or sys.argv[i] == '-model':
            params['modelFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-c' or sys.argv[i] == '-config':
            params['configFilename'] = sys.argv[i + 1]

    if params['configFilename'] is None:
        raise Exception('Config file not specified')
    if params['trainDataFilename'] is None:
        raise Exception('Data file not specified')
    if params['modelFilename'] is None:
        raise Exception('Model file not specified')

    return params

if __name__ == '__main__':
    params = readFlags()
    with open(params['configFilename']) as f:
        trainOpt = yaml.load(f)
    trainData = np.load(params['trainDataFilename'])
    trainInput = trainData[0]
    trainTarget = trainData[1]
    if params['validDataFilename'] is not None:
        validData = np.load(params['validDataFilename'])
        validInput = validData[0]
        validTarget = validData[1]
    else:
        validInput = None
        validTarget = None
    model = nn.load(params['modelFilename'])
    trainer = nn.Trainer(
        name=params['name']+'-v',
        model=model,
        trainOpt=trainOpt,
        outputFolder=params['outputFolder']
    )

    trainer.train(trainInput, trainTarget, validInput, validTarget)
    # Send email
    if trainOpt.has_key('sendEmail') and trainOpt['sendEmail']:
        email.appendList(params['outputFolder'], trainer.name)

    if params['testDataFilename'] is not None:
        testData = np.load(params['testDataFilename'])
        testInput = testData[0]
        testTarget = testData[1]
        testOutput = nn.test(model, testInput)
        testRate, c, t = nn.calcRate(model, testOutput, testTarget)
        print 'Before retrain test rate: ', testRate

        # Retrain with all the data
        trainOpt['needValid'] = False
        trainOpt['numEpoch'] = trainer.stoppedEpoch + 1
        model = nn.load(params['modelFilename'])
        trainer = nn.Trainer(
            name=params['name'],
            model=model,
            trainOpt=trainOpt,
            outputFolder=params['outputFolder']
        )

        if params['allDataFilename'] is not None:
            allData = np.load(params['allDataFilename'])
            allInput = allData[0]
            allTarget = allData[1]
            trainer.train(allInput, allTarget)
        else:
            allInput = np.concatenate((trainInput, validInput), axis=0)
            allTarget = np.concatenate((trainTarget, validTarget), axis=0)
            trainer.train(allInput, allTarget)

        model = nn.load(params['modelFilename'])
        model.loadWeights(np.load(trainer.modelFilename))
        testOutput = nn.test(model, testInput)
        testRate, c, t = nn.calcRate(model, testOutput, testTarget)
        print 'After retrain test rate: ', testRate

        with open(os.path.join(trainer.outputFolder, 'result.txt'), 'w+') as f:
            f.write('Test rate: %f' % testRate)
        # Send email
        if trainOpt.has_key('sendEmail') and trainOpt['sendEmail']:
            email.appendList(params['outputFolder'], trainer.name)