import json
import os
import cPickle as pkl

if __name__ == '__main__':
    with open('../../../data/mscoco/captions_train2014.json') as f:
        captiontxt = f.read()
    caption = json.loads(captiontxt)
    L = len(caption['annotations'])

    dataset = []
    for i in range(L):
        sent = caption['annotations'][i]['caption']
        imgid = caption['annotations'][i]['image_id']

        sent = sent.replace('\n','').strip()
        dataset.append((sent, imgid))
        #if sent.startswith('One motorcycle witha large'):
        #print (sent, imgid)

    with open('../../../data/mscoco/mscoco_caption.txt', 'w') as f:
        for d in dataset:
            f.write(d[0] + '\n')
    with open('../../../data/mscoco/mscoco_imgids.txt', 'w') as f:
        for d in dataset:
            f.write(str(d[1]) + '\n')
    # with open('../../../data/mscoco/mscoco_caption.pkl', 'wb') as f:
    #     pkl.dump(dataset, f)
