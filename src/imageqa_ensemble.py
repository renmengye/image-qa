import sys
import nn
import numpy as np
from imageqa_test import *

if __name__ == '__main__':
    """
    Test a type-specific ensemble model
    Usage:
    python imageqa_ensemble.py -i[ds] {taskId1},{taskId2},...
                               -d[ata] {dataFolder}
                               [-r[esults] {resultsFolder}]
    Results folder by default is '../results'
    """
    resultsFolder = '../results'
    for i, flag in enumerate(sys.argv):
        if flag == '-i' or flag == '-ids':
            taskIds = sys.argv[i + 1].split(',')
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == 'r' or flag == 'results':
            resultsFolder = sys.argv[i + 1]
    models = []
    for taskId in taskIds:
        taskFolder = os.path.join(resultsFolder, taskId)
        modelSpec = os.path.join(resultsFolder, '%s.model.yml' % taskId)
        modelWeights = os.path.join(resultsFolder, '%s.w.npy' % taskId)
        model = nn.load(modelSpec)
        model.loadWeights(np.load(modelWeights))
        models.append(model)

    imageqa_test.testEnsemble(
                                ensembleId='ensemble',
                                taskIds=taskIds, 
                                models=models, 
                                dataFolder=dataFolder, 
                                resultsFolder=resultsFolder)