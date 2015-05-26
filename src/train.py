import nn
import numpy as np
import os
import sys
import yaml
import experiment_email as email
import imageqa_test

"""
Train a neural network
Usage: python train.py
                    -n[ame] {name} 
                    -d[ata] {train/valid/test folder}
                    -m[odel] {model spec} 
                    -c[onfig] {config filename}
                    -w[eights] {input weights}
                    -o[utput] {output folder}
                    -b[oard] {board id}
                    [-imageqa]
Prameters: 
    -n[ame] Name of the model 
    -d[ata] Data folder that contains 'train.npy', 'valid.npy', and 'test.npy'
    -m[odel] Model specification file name
    -c[onfig] Train config file name
    -w[eights] Weighted input (for boosting), filename before '-train' or '-test'
    -o[utput] Training results output folder
    -b[oard] GPU board ID
    [-imageqa] Run Image-QA test scripts
"""

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
    params['trainInputWeightsFilename'] = None
    params['validInputWeightsFilename'] = None
    for i, flag in enumerate(sys.argv):
        if flag == '-n' or flag == '-name':
            params['name'] = sys.argv[i + 1]
        elif flag == '-o' or flag == '-out':
            params['outputFolder'] = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
            trainPath = os.path.join(dataFolder, 'train.npy')
            params['trainDataFilename'] = trainPath if os.path.isfile(trainPath) else None
            validPath = os.path.join(dataFolder, 'valid.npy')
            params['validDataFilename'] = validPath if os.path.isfile(validPath) else None
            testPath = os.path.join(dataFolder, 'test.npy')
            params['testDataFilename'] = testPath if os.path.isfile(testPath) else None
            params['dataFolder'] = dataFolder
        elif flag == '-w' or flag == '-weights':
            weightsPath = sys.argv[i + 1]
            params['trainInputWeightsFilename'] = os.path.join(dataFolder, weightsPath + '-train.npy')
            params['validInputWeightsFilename'] = os.path.join(dataFolder, weightsPath + '-valid.npy')
            params['testInputWeightsFilename'] = os.path.join(dataFolder, weightsPath + '-test.npy')
        elif flag == '-m' or flag == '-model':
            params['modelFilename'] = sys.argv[i + 1]
        elif flag == '-c' or flag == '-config':
            params['configFilename'] = sys.argv[i + 1]
        elif flag == '-b' or flag == '-board':
            os.environ['GNUMPY_BOARD_ID'] = sys.argv[i + 1]
        elif flag == '-imageqa':
            params['imageqa'] = True
        elif flag == '-noimageqa':
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

    if params['trainInputWeightsFilename'] is not None:
        trainInputWeights = np.load(params['trainInputWeightsFilename'])
        validInputWeights = np.load(params['validInputWeightsFilename'])
    else:
        trainInputWeights = None
        validInputWeights = None  

    model = nn.load(params['modelFilename'])
    trainer = nn.Trainer(
        name=params['name']+\
        ('-v' if params['validDataFilename'] is not None else ''),
        model=model,
        trainOpt=trainOpt,
        outputFolder=params['outputFolder']
    )

    trainer.train(
                trainInput=trainInput, 
                trainTarget=trainTarget,
                trainInputWeights=trainInputWeights,
                validInput=validInput, 
                validTarget=validTarget,
                validInputWeights=validInputWeights)
    
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

        allInput = np.concatenate((trainInput, validInput), axis=0)
        allTarget = np.concatenate((trainTarget, validTarget), axis=0)
        if trainInputWeights is not None:
            allInputWeights = np.concatenate(
                (trainInputWeights, validInputWeights), axis=0)
        else:
            allInputWeights = None
        trainer.train(
                trainInput=allInput, 
                trainTarget=allTarget,
                trainInputWeights=allInputWeights)

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