import yaml
import router

def load(modelSpecFilename):
    with open(modelSpecFilename) as f:
        modelDict = yaml.load(f)

    for stageDict in modelDict['stages']:
        router.routeStage(stageDict)

    costFn=router.routeFn(modelDict['costFn'])
    decisionFn=router.routeFn(modelDict['decisionFn'])
    model = router.getStage(modelDict['model'])
    model.getCost = costFn
    model.predict = decisionFn
    model.specFilename = modelSpecFilename

    return model