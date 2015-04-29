import nn
import numpy as np
import os
import sys
import yaml
import experiment_email as email
import imageqa_test

'''
Usage: python train.py {name} -d {train/valid/test folder}
                              -m {model spec} 
                              -c {config} 
                              -o {output folder}
                              [-imageqa]
'''

def readFlags():
    params = {}
    params['name'] = None
    params['outputFolder'] = None
    params['configFilename'] = None
    params['trainDataFilename'] = None
    params['testDataFilename'] = None
    params['validDataFilename'] = None
    params['allDataFilename'] = None
    params['modelFilename'] = None
    params['imageqa'] = True
    for i in range(1, len(sys.argv) - 1):
        if sys.argv[i] == '-n' or sys.argv[i] == '-name':
            params['name'] == sys.argv[i + 1]
        elif sys.argv[i] == '-o' or sys.argv[i] == '-out':
            params['outputFolder'] = sys.argv[i + 1]
        elif sys.argv[i] == '-d' or sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
            trainPath = os.path.join(dataFolder, 'train.npy')
            params['trainDataFilename'] = trainPath if os.path.isfile(trainPath) else None
            validPath = os.path.join(dataFolder, 'valid.npy')
            params['validDataFilename'] = validPath if os.path.isfile(validPath) else None
            testPath = os.path.join(dataFolder, 'test.npy')
            params['testDataFilename'] = testPath if os.path.isfile(testPath) else None
            params['dataFolder'] = dataFolder
        elif sys.argv[i] == '-a' or sys.argv[i] == '-alldata':
            params['allDataFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-m' or sys.argv[i] == '-model':
            params['modelFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-c' or sys.argv[i] == '-config':
            params['configFilename'] = sys.argv[i + 1]
        elif sys.argv[i] == '-imageqa':
            params['imageqa'] = True
        elif sys.argv[i] == '-noimageqa':
            params['imageqa'] = False

    # Check required parameters.
    if params['configFilename'] is None:
        raise Exception('Config file not specified')
    if params['trainDataFilename'] is None:
        raise Exception('Data file not specified')
    if params['modelFilename'] is None:
        raise Exception('Model file not specified')
    if params['name'] is None:
        params['name'] = params['modelFilename'].split('/')[-1].split('.')[0]

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
        name=params['name']+\
        ('-v' if params['validDataFilename'] is not None else ''),
        model=model,
        trainOpt=trainOpt,
        outputFolder=params['outputFolder']
    )

    trainer.train(trainInput, trainTarget, validInput, validTarget)
    
    if params['testDataFilename'] is not None:
        if params['imageqa']:
            imageqa_test.testAll(
                trainer.name, model, params['dataFolder'], params['outputFolder'])
        else:
            testData = np.load(params['testDataFilename'])
            testInput = testData[0]
            testTarget = testData[1]
            model.loadWeights(np.load(trainer.modelFilename))
            testOutput = nn.test(model, testInput)
            testRate, c, t = nn.calcRate(model, testOutput, testTarget)
            print 'Test rate: ', testRate
            with open(os.path.join(
                trainer.outputFolder, 'result.txt'), 'w+') as f:
                f.write('Test rate: %f\n' % testRate)
    
    # Send email
    if trainOpt.has_key('sendEmail') and trainOpt['sendEmail']:
        email.appendList(params['outputFolder'], trainer.name)

    if params['testDataFilename'] is not None and\
        params['validDataFilename'] is not None:
        # Retrain with all the data
        trainOpt['needValid'] = False
        print 'Stopped score:', trainer.stoppedTrainScore
        trainOpt['stopScore'] = trainer.stoppedTrainScore
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
        if params['imageqa']:
            imageqa_test.testAll(
                trainer.name, model, params['dataFolder'], params['outputFolder'])
        else:
            testOutput = nn.test(model, testInput)
            testRate, c, t = nn.calcRate(model, testOutput, testTarget)
            print 'Test rate: ', testRate
            with open(os.path.join(
                trainer.outputFolder, 'result.txt'), 'w+') as f:
                f.write('Test rate: %f' % testRate)
        
        # Send email
        if trainOpt.has_key('sendEmail') and trainOpt['sendEmail']:
            email.appendList(params['outputFolder'], trainer.name)
