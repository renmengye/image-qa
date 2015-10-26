import sys
import numpy

if __name__ == '__main__':
    gtFile = sys.argv[1]
    outFile = sys.argv[2]
    qtypeFile = sys.argv[3]
    qtype = numpy.load(qtypeFile)
    with open(gtFile) as f:
        gt = f.readlines()
    with open(outFile) as f:
        out = f.readlines()

    count = numpy.zeros(4)
    total = numpy.zeros(4)

    for i, (gt_i, out_i) in enumerate(zip(gt, out)):
        if gt_i == out_i:
            count[qtype[i]] += 1
        total[qtype[i]] += 1

    print count /total
