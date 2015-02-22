import yaml
import router
from model import *

def load(modelSpecFilename):
    with open(modelSpecFilename) as f:
        modelDict = yaml.load(f)

    for stageDict in modelDict['stages']:
        router.routeStage(stageDict)

    costFn=router.routeFn(modelDict['costFn'])
    decisionFn=router.routeFn(modelDict['decisionFn'])
    model = Model(router.getStage(modelDict['model']),
        costFn=costFn,
        decisionFn=decisionFn,
        specFilename=modelSpecFilename)

    return model