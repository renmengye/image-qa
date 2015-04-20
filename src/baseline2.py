import sys

if len(sys.argv) < 6:
    colorAnswer = 'white'
    numberAnswer = 'two'
    objectAnswer = 'table'
    gtFile = '../gt.txt'
    outFile = '../guess.txt'
else:
    colorAnswer = sys.argv[1]
    numberAnswer = sys.argv[2]
    objectAnswer = sys.argv[3]
    gtFile = sys.argv[4]
    outFile = sys.argv[5]

with open(gtFile) as f:
    gt = f.readlines()

correct = 0
total = len(gt)
with open(outFile, 'w') as f:
    for item in gt:
        answer = item[:-1]
        if answer == 'one' or answer == 'two' or answer == 'three' or\
            answer == 'four' or answer == 'five' or answer == 'six' or\
            answer == 'seven' or answer == 'eight' or answer == 'nine' or\
            answer == 'ten' or answer == 'eleven' or answer == 'twelve' or\
            answer == 'thirteen' or answer == 'fourteen' or answer == 'fifteen' or\
            answer == 'sixteen' or answer == 'seventeen' or answer == 'eighteen' or\
            answer == 'nineteen' or answer == 'twenty' or answer == 'twenty-one' or\
            answer == 'twenty-two' or answer == 'twenty-three' or answer == 'twenty-four' or\
            answer == 'twenty-five' or answer == 'twenty-six' or answer == 'twenty-seven':
            f.write(numberAnswer + '\n')
            if answer == numberAnswer: correct += 1
        elif answer == 'red' or answer == 'orange' or answer == 'yellow' or\
            answer == 'green' or answer == 'blue' or answer == 'black' or\
            answer == 'white' or answer == 'brown' or answer == 'grey' or\
            answer == 'gray' or answer == 'purple' or answer == 'pink':
            f.write(colorAnswer + '\n')
            if answer == colorAnswer: correct += 1
        else:
            f.write(objectAnswer + '\n')
            if answer == objectAnswer: correct += 1
print correct / float(total)