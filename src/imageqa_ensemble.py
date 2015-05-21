import sys
import os
import nn
import numpy as np
import imageqa_test as it
import imageqa_visprior as ip

def runAllModels(
                inputTest, 
                questionTypeArray, 
                modelSpecs, 
                resultsFolder):
    allOutputs = []
    for modelSpec in modelSpecs:
        if modelSpec['isEnsemble']:
            print 'Running test data on ensemble model %s...' \
                    % modelSpec['name']
            models = it.loadEnsemble(modelSpec['id'].split(','), resultsFolder)
            classDataFolders = it.getClassDataFolder(dataset, dataFolder)
            if modelSpec['runPrior']:
                outputTest = ip.runEnsemblePrior(
                                        inputTest, 
                                        models,
                                        dataFolder,
                                        classDataFolders,
                                        questionTypeArray)
            else:
                outputTest = it.runEnsemble(
                                        inputTest, 
                                        models,
                                        dataFolder,
                                        classDataFolders,
                                        questionTypeArray)
        else:
            print 'Running test data on model %s...' \
                    % modelSpec['name']
            model = it.loadModel(modelId, resultsFolder)
            outputTest = nn.test(model, inputTest)
        allOutputs.append(outputTest)
    return allOutputs

if __name__ == '__main__':
    """
    Test a type-specific ensemble model
    Usage:
    python imageqa_ensemble.py -e[nsemble] {ensembleId}
                               -m[odel] {modelId1}
                               -m[odel] {modelId2},...
                               -d[ata] {dataFolder}
                               -dataset {daquar/cocoqa}
                               [-r[esults] {resultsFolder}]
                               [-prior]
    Results folder by default is '../results'
    """
    resultsFolder = '../results'
    taskIds = []
    dataset = 'cocoqa'
    runPrior = False
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            taskIds.append(sys.argv[i + 1])
        elif flag == '-e' or flag == '-ensemble':
            ensembleId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-dataset':
            dataset = sys.argv[i + 1]
        elif flag == '-prior':
            runPrior = True
    models = it.loadEnsemble(taskIds, resultsFolder)
    classDataFolders = it.getClassDataFolders(dataset, dataFolder)
    if runPrior:
        ip.testEnsemblePrior(
                        ensembleId=ensembleId,
                        models=models, 
                        dataFolder=dataFolder, 
                        classDataFolders=classDataFolders,
                        resultsFolder=resultsFolder)
    else:
        it.testEnsemble(
                        ensembleId=ensembleId,
                        models=models, 
                        dataFolder=dataFolder, 
                        classDataFolders=classDataFolders,
                        resultsFolder=resultsFolder)