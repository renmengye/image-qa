import sys
import os
import nn
import numpy as np
import imageqa_test as it
import imageqa_visprior as ip
import imageqa_render as ir

if __name__ == '__main__':
    """
    Render a selection of examples into LaTeX.
    Usage: python imageqa_layout.py 
                    -m[odel] {name1:modelId1}
                    -m[odel] {name2:modelId2}
                    -em[odel] {name3:ensembleModelId3,ensembleModelId4,...}
                    -pem[odel] {name4:ensembleModelId5,ensembleModelId6,...}
                    ...
                    -d[ata] {dataFolder}
                    -i[nput] {listFile}
                    -o[utput] {outputFolder}
                    [-k {top K answers}]
                    [-p[icture] {pictureFolder}]
                    [-f[ile] {outputTexFilename}]
                    [-daquar/-cocoqa]
    Parameters:
        -m[odel]
        -d[ata]
        -i[nput]
        -k
        -p[icture]
        -f[ile]
        -daquar
        -cocoqa
    Input file format:
    QID1[,Comment1]
    QID2[,Comment2]
    ...
    """
    params = ir.parseComparativeParams(sys.argv)
    dataset = params['dataset']
    dataFolder = params['dataFolder']

    print 'Loading image urls...'
    if dataset == 'cocoqa':
        imgidDictFilename = os.path.join(dataFolder, 'imgid_dict.pkl')
        with open(imgidDictFilename, 'rb') as f:
            imgidDict = pkl.load(f)
        urlDict = readImgDictCocoqa(imgidDict)
    elif dataset == 'daquar':
        urlDict = readImgDictDaquar()

    print 'Loading test data...'
    
    selectionIds = []
    selectionComments = []
    caption = ''
    with open(inputFile) as f:
        i = 0
        for line in f:
            if i == 0 and line.startswith('caption:'):
                caption = line[8:-1]
            else:
                parts = line.split(',')
                selectionIds.append(int(parts[0]))
                if len(parts) > 1:
                    selectionComments.append(parts[1][:-1])
                else:
                    selectionComments.append('')
            i += 1
    idx = np.array(selectionIds, dtype='int')
    inputTestSel = inputTest[idx]
    targetTestSel = targetTest[idx]
    
    for i in range(len(questionIdict)):
        if '_' in questionIdict[i]:
            questionIdict[i] = questionIdict[i].replace('_', '\\_')
    for i in range(len(ansIdict)):
        if '_' in ansIdict[i]:
            ansIdict[i] = ansIdict[i].replace('_', '\\_')

    modelOutputs = []
    for rp, modelName, modelId in zip(runPriors, modelNames, modelIds):
        if ',' in modelId:
            print 'Running test data on ensemble model %s...' \
                    % modelName
            models = it.loadEnsemble(modelId.split(','), resultsFolder)
            classDataFolders = it.getClassDataFolder(dataset, dataFolder)
            if rp:
                outputTest = ip.runEnsemblePrior(
                                        adhocInputTestSel, 
                                        models,
                                        dataFolder,
                                        classDataFolders,
                                        adhocQuestionTypeTestSel)
            else:
                outputTest = it.runEnsemble(
                                        adhocInputTestSel, 
                                        models,
                                        dataFolder,
                                        classDataFolders,
                                        adhocQuestionTypeTestSel)
        else:
            print 'Running test data on model %s...' \
                    % modelName
            model = it.loadModel(modelId, resultsFolder)
            outputTest = nn.test(model, inputTest)
        modelOutputs.append(outputTest)

    # Render
    print('Rendering LaTeX...')
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    ir.renderLatex(
                inputTestSel, 
                targetTestSel, 
                questionIdict, 
                ansIdict, 
                urlDict, 
                topK=K,
                outputFolder=outputFolder,
                pictureFolder=pictureFolder,
                comments=selectionComments,
                caption=caption,
                modelOutputs=modelOutputs,
                modelNames=modelNames,
                questionIds=idx,
                filename=filename + '.tex')