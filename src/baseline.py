import sys
import numpy as np

outFile = sys.argv[1]
gtFile = sys.argv[2]

with open(gtFile) as f:
    gt = f.readlines()
with open(outFile) as f:
    out = f.readlines()

colorAnswer = 'white'
numberAnswer = 'two'
objectAnswer = 'cat'
locationAnswer = 'room'

correct = np.zeros(4)
total = np.zeros(4)

for item in zip(out, gt):
    answer = item[0]
    if answer.startswith(objectAnswer):
        if answer == item[1]: correct[0] += 1
        total[0] += 1
    elif answer.startswith(numberAnswer):
        if answer == item[1]: correct[1] += 1
        total[1] += 1
    elif answer.startswith(colorAnswer):
        if answer == item[1]: correct[2] += 1
        total[2] += 1
    elif answer.startswith(locationAnswer):
        if answer == item[1]: correct[3] += 1
        total[3] += 1

print correct, total, correct / total.astype(float)