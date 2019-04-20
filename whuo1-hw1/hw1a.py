import argparse
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class EOSClassifier:
    def train(self, trainX, trainY):
        self.abbrevs = load_wordlist('classes/abbrevs')

        self.clf = DecisionTreeClassifier()
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, array):
        id, word_m3, word_m2, word_m1, period, word_p1, word_p2, word_3, left_reliable, right_reliable, num_spaces = array
        features = [
            left_reliable,
            right_reliable,
            num_spaces,
            1 if word_m1 in self.abbrevs else 0,
            # len(word_m1),
            # 1 if word_p1.isupper() else 0,
        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)

class LogisticClassifier:
    def train(self, trainX, trainY):
        self.clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1000)
        X = get_features(trainX)
        self.clf.fit(X, trainY)

    def classify(self, testX):
        X = get_features(testX)
        return self.clf.predict(X)

class SVMClassifier:
    def train(self, trainX, trainY):
        self.clf = SVC(kernel='rbf', C=1, gamma='scale')
        X = get_features(trainX)
        self.clf.fit(X, trainY)

    def classify(self, testX):
        X = get_features(testX)
        return self.clf.predict(X)

def load_wordlist(file):
    with open(file) as fin:
        return set([x.strip() for x in fin.readlines()])

def load_part_of_speech(file):
    res = dict()
    types = []
    with open(file) as fin:
        for line in fin:
            arr = line.strip().split()
            if arr[2] not in types:
                types.append(arr[2])
        for line in fin:
            arr = line.strip().split()
            res[arr[1]] = [int(arr[0]), types.index(arr[2])]
    return res

def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split()
            X.append(arr[1:])
            y.append(arr[0])
        return X, y

def get_features(X):
    abbrevs = load_wordlist('classes/abbrevs')
    sentence_internal = load_wordlist('classes/sentence_internal')
    timeterms = load_wordlist('classes/timeterms')
    titles = load_wordlist('classes/titles')
    unlikely_proper_nouns = load_wordlist('classes/unlikely_proper_nouns')
    histogram = load_part_of_speech('data/part-of-speech.histogram')
    def convert(arr):
        res = []
        for i in range(len(arr)):
            # if it is digits, keep it
            element = arr[i]
            if i in [0, 8, 9, 10]:
                res.append(int(element))
            # if it is word, convert to more features
            else:
                res.append(1 if re.match(r'[^a-zA-Z0-9]+', element) else 0)
                res.append(1 if element in abbrevs else 0)
                res.append(1 if element in sentence_internal else 0)
                res.append(1 if element in timeterms else 0)
                res.append(1 if element in titles else 0)
                res.append(1 if element in unlikely_proper_nouns else 0)
                res += (histogram[element] if element in histogram else [0, len(histogram)])
                # new paragraph
                res.append(1 if re.match(r'<P>', element) else 0)
                # special characters
                res.append(1 if re.match(r'[\.,;-]', element) else 0)
                # double quote
                res.append(1 if re.match(r'``', element) else 0)
                # all digits
                res.append(1 if re.match(r'[0-9]+', element) else 0)
                # uppercase character
                res.append(1 if re.match(r'^[A-Z]$', element) else 0)

        return res
    X = list(map(convert, X))
    ss = StandardScaler()
    X = ss.fit_transform(X) 
    return X


def evaluate(outputs, golds):
    correct = 0
    for h, y in zip(outputs, golds):
        if h == y:
            correct += 1
    print(f'{correct} / {len(golds)}  {correct / len(golds)}')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()
    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)
    
    # for development (need to comment out if not for development)
    # begin
    #X, y = load_data(args.train)
    #trainX, testX, trainY, testY = train_test_split(X,y,test_size=0.2)
    # end

    #classifier = EOSClassifier()
    #classifier = LogisticClassifier()
    classifier = SVMClassifier()
    classifier.train(trainX, trainY)
    outputs = classifier.classify(testX)

    if args.output is not None:
        with open(args.output, 'w') as fout:
            for output in outputs:
                print(output, file=fout)

    if args.errors is not None:
        with open(args.errors, 'w') as fout:
            for y, h, x in zip(testY, outputs, testX):
                if y != h:
                    print(y, h, x, sep='\t', file=fout)

    if args.report:
        print(classification_report(testY, outputs))
    else:
        evaluate(outputs, testY)


if __name__ == '__main__':
    main()