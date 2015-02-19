import json
import os

if __name__ == '__main__':
    with open('../../../data/mscoco/captions_train2014.json') as f:
        captiontxt = f.read()
    caption = json.loads(captiontxt)
    L = len(caption['annotations'])

    with open('../../../data/mscoco/mscoco_caption.txt', 'w+') as f:
        for i in range(L):
            sent = caption['annotations'][i]['caption']
            if not '.' in sent:
                sent = sent + '.'
            f.write(sent + '\n')

    with open('../../../data/mscoco/mscoco_imgid.txt', 'w+') as f:
        for i in range(L):
            f.write('%d\n' % caption['annotations'][i]['image_id'])
