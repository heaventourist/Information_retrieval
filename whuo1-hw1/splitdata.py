import random
import sys

import hw1b

random.seed(12345)

if sys.argv[1] == 'sent':
    with open('sent.train') as fin:
        lines = [x.rstrip('\n') for x in fin.readlines()]

    random.shuffle(lines)

    with open('sent.train1', 'w') as train, open('sent.dev1', 'w') as dev:
        split = int(len(lines) * 0.9)
        for line in lines[:split]:
            print(line, file=train)
        for line in lines[split:]:
            print(line, file=dev)

else:
    X, y = hw1b.load_data('segment.train')
    segX, segY = hw1b.lines2segments(X, y)
    data = list(zip(segX, segY))

    random.shuffle(data)

    with open('segment.train1', 'w') as train, open('segment.dev1', 'w') as dev:
        split = int(len(data) * 0.9)
        for segX, segY in data[:split]:
            for line in segX.split('\n'):
                print(segY, line, sep='\t', file=train)
        for segX, segY in data[split:]:
            for line in segX.split('\n'):
                print(segY, line, sep='\t', file=dev)