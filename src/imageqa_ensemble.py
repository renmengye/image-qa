import sys
import nn
import numpy as np
from imageqa_test import *

if __name__ == '__main__':
    """
    Test a type-specific ensemble model
    Usage:
    python imageqa_ensemble.py -e[nsemble] {ensembleId}
                               -m[odel] {modelId1}
                               -m[odel] {modelId2},...
                               -d[ata] {dataFolder}
                               [-r[esults] {resultsFolder}]
    Results folder by default is '../results'
    """
    resultsFolder = '../results'
    taskIds = []
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            taskIds.append(sys.argv[i + 1])
        elif flag == '-e' or flag == '-ensemble':
            ensembleId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == 'r' or flag == 'results':
            resultsFolder = sys.argv[i + 1]
    models = loadEnsemble(taskIds)
    testEnsemble(
                    ensembleId=ensembleId,
                    models=models, 
                    dataFolder=dataFolder, 
                    resultsFolder=resultsFolder)