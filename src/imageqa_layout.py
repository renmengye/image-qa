import sys
import os
import nn
import numpy as np
import imageqa_test as it
import imageqa_visprior as ip
import imageqa_ensemble as ie
import imageqa_render as ir

def parseCommentsFile(filename):
    caption = ''
    selIds = []
    selComments = []
    with open(filename) as f:
        i = 0
        for line in f:
            if i == 0 and line.startswith('caption:'):
                caption = line[8:-1]
            else:
                parts = line.split(',')
                selIds.append(int(parts[0]))
                if len(parts) > 1:
                    selComments.append(parts[1][:-1])
                else:
                    selComments.append('')
            i += 1
    return caption, selIds, selComments

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
    inputFile = params['inputFile']
    data = it.loadDataset(params['dataFolder'])

    print 'Loading image urls...'
    if dataset == 'cocoqa':
        imgidDictFilename = os.path.join(dataFolder, 'imgid_dict.pkl')
        with open(imgidDictFilename, 'rb') as f:
            imgidDict = pkl.load(f)
        urlDict = readImgDictCocoqa(imgidDict)
    elif dataset == 'daquar':
        urlDict = readImgDictDaquar()

    print 'Loading test data...'

    caption, selIds, selComments = parseCommentsFile(inputFile)

    idx = np.array(selIds, dtype='int')
    inputTestSel = inputTest[idx]
    targetTestSel = targetTest[idx]

    inputTest = data['testData'][0]
    questionTypeArray = data['questionTypeArray']
    modelSpecs = params['models']
    modelOutputs = ie.runAllModels(
                inputTestSel, 
                questionTypeArray[idx], 
                modelSpecs, 
                resultsFolder):

    # Render
    print('Rendering LaTeX...')
    
    # Replace escape char
    questionIdict = data['questionIdict']
    ansIdict = data['ansIdict']
    for i in range(len(questionIdict)):
        if '_' in questionIdict[i]:
            questionIdict[i] = questionIdict[i].replace('_', '\\_')
    for i in range(len(ansIdict)):
        if '_' in ansIdict[i]:
            ansIdict[i] = ansIdict[i].replace('_', '\\_')

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    modelNames = []
    for spec in modelSpecs:
        modelNames.append(spec['name'])
    ir.renderLatex(
                inputTestSel, 
                targetTestSel, 
                data['questionIdict'], 
                data['ansIdict'],
                urlDict, 
                topK=params['topK'],
                outputFolder=params['outputFolder'],
                pictureFolder=params['pictureFolder'],
                comments=selComments,
                caption=caption,
                modelOutputs=modelOutputs,
                modelNames=modelNames,
                questionIds=idx,
                filename=filename + '.tex')