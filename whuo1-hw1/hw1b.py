import argparse
from itertools import groupby
import re

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SegmentClassifier:
    def train(self, trainX, trainY):
        self.clf = DecisionTreeClassifier()
        X = [self.extract_features(x) for x in trainX]
        self.clf.fit(X, trainY)

    def extract_features(self, text):
        words = text.split()
        features = [
            len(text),
            len(text.strip()),
            len(words),
            1 if '>' in words else 0,
            # text.count(' '),
            # sum(1 if w.isupper() else 0 for w in words)
        ]
        return features

    def classify(self, testX):
        X = [self.extract_features(x) for x in testX]
        return self.clf.predict(X)

class SVMClassifier:
    def train(self, trainX, trainY):
        self.clf = SVC(kernel='rbf', C=1, gamma='scale')
        X = [get_features(x) for x in trainX]
        ss = StandardScaler()
        X = ss.fit_transform(X)
        self.clf.fit(X, trainY)

    def classify(self, testX):
        X = [get_features(x) for x in testX]
        ss = StandardScaler()
        X = ss.fit_transform(X)
        return self.clf.predict(X)

class LogisticClassifier:
    def train(self, trainX, trainY):
        self.clf = LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=1000)
        X = [get_features(x) for x in trainX]
        ss = StandardScaler()
        X = ss.fit_transform(X) 
        self.clf.fit(X, trainY)

    def classify(self, testX):
        X = [get_features(x) for x in testX]
        ss = StandardScaler()
        X = ss.fit_transform(X) 
        return self.clf.predict(X)

def load_data(file):
    with open(file) as fin:
        X = []
        y = []
        for line in fin:
            arr = line.strip().split('\t', 1)
            if arr[0] == '#BLANK#':
                continue
            X.append(arr[1])
            y.append(arr[0])
        return X, y


def lines2segments(trainX, trainY):
    segX = []
    segY = []
    for y, group in groupby(zip(trainX, trainY), key=lambda x: x[1]):
        if y == '#BLANK#':
            continue
        x = '\n'.join(line[0].rstrip('\n') for line in group)
        segX.append(x)
        segY.append(y)
    return segX, segY


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
    parser.add_argument('--format', required=True)
    parser.add_argument('--output')
    parser.add_argument('--errors')
    parser.add_argument('--report', action='store_true')
    return parser.parse_args()


def main():
    args = parseargs()

    trainX, trainY = load_data(args.train)
    testX, testY = load_data(args.test)

    if args.format == 'segment':
        trainX, trainY = lines2segments(trainX, trainY)
        testX, testY = lines2segments(testX, testY)

    # for development (need to comment out if not for development)
    # begin
    #X, y = load_data(args.train)
    #trainX, testX, trainY, testY = train_test_split(X,y,test_size=0.2)

    #if args.format == 'segment':
        #X, y = lines2segments(X, y)
        #trainX, testX, trainY, testY = train_test_split(X,y,test_size=0.2)
    # end
    
    #classifier = SegmentClassifier()
    classifier = SVMClassifier()
    #classifier = LogisticClassifier()
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

def get_features(text):
    text = text.split('\n')
    return [head_detector(text), quoted_detector(text), table_detector(text), item_detector(text),
    sig_detector(text), graphic_detector(text), uppercase_ratio(text), begin_whitespace(text),
    continuous_special_char(text), num_periods(text), white_space_ratio(text), begin_alphabet(text),
    address_detector(text), same_indent(text), text_detector(text)]

############################################################
## 'NNHEAD' detector
############################################################
def head_detector(text):
    score = 0
    regex = r'^From:|^Article:|^Path:|^Newsgroups:|^Subject:|^Date:|^Organization:|^Lines:|^Approved:|^Message-ID:|^References:|^NNTP-Posting-Host:'
    for line in text:
        if re.match(regex, line):
            score += 1
    return score

############################################################
## 'QUOTED' detector
############################################################
def quoted_detector(text):
    score = 0

    test_quote1 = r'^(>|:|\s*\S*\s*>|@)(\s*(>|:))*'
    test_quote2 = r'^.+(wrote|writes|said|post):'

    for line in text:
        if re.match(test_quote1, line) or re.match(test_quote2, line):
            score += 1
    return score

############################################################
## 'SIG' detector
############################################################
def sig_detector(text):
    if re.match(r'^--\s*', text[0]):
        return 1
    elif len(text) <= 10:
        # Assumes there are no 'SIG's that exceed 10 lines
        threshold = 10
        normal_char = r'[a-zA-Z0-9\s]+'

        for line in text:
            continuous_special_chars = 0
            chars = list(line)
            for c in chars:
                if re.match(normal_char, c):
                    continuous_special_chars = 0
                else:
                    continuous_special_chars += 1
                if continuous_special_chars > threshold:
                    return 1
        return 0
    else:
        return 0

############################################################
## Returns the maximum number of repeated special characters
############################################################
def continuous_special_char(text):
    normal_char = r'[a-zA-Z0-9\s]+'
    
    max_continuous_special_char = 0

    for line in text:
        continuous_special_char = 0
        chars = list(line)
        for c in chars:
            if re.match(normal_char, c):
                max_continuous_special_char = max(max_continuous_special_char, continuous_special_char)
                continuous_special_char = 0
            else:
                continuous_special_char += 1
    return max_continuous_special_char

############################################################
## 'TABLE' detector
############################################################
def table_detector(text):
    if len(text) <= 2:
        return 0
    score = 0
    prev_word_count = -1

    # Check if there are more than two lines with "more or less" same number of words
    for line in text:
        words = line.split()
        word_count = len(words)

        if prev_word_count == -1:
            prev_word_count = word_count
        elif prev_word_count - 1 < word_count and word_count < prev_word_count + 1:
            score += 1
        else:
            score = 0
            prev_word_count = -1
        if score > 2:
            return 1
    return 0

############################################################
## 'GRAPHIC' detector
############################################################
def graphic_detector(text):
    letter_count = 0
    non_letter_count = 0

    normal_char = r'[a-zA-Z0-9\s]+'

    for line in text:
        chars = list(line)
        for c in chars:
            if re.match(normal_char, c):
                letter_count += 1
            else:
                non_letter_count += 1
    return non_letter_count / (letter_count + 0.00001)

############################################################
## Ratio between number of words that start with upper case
## and total number of words
############################################################
def uppercase_ratio(text):
    num_cap_words = 0
    num_words = 0
    upper_case = r'^[0-9A-Z]'

    for line in text:
        words = line.split()
        for w in words:
            # Count how many words start in upper case or digit
            if re.match(upper_case, w):
                num_cap_words += 1
            num_words += 1

    return num_cap_words / num_words

############################################################
## Checks whether every line begins with a whitespace
############################################################
def begin_whitespace(text):
    num_white_lines = 0
    white_space = r'[\s]{5, 1000}.*'

    for line in text:
        # Check how many lines start with a "large" white space.
        if re.match(white_space, line):
            num_white_lines += 1
    if len(text) == num_white_lines:
        # If every line starts with a large white space...
        return 1
    else:
        return 0

############################################################
## Counts the number of period marks
############################################################
def num_periods(text):
    num_periods = 0

    for line in text:
        chars = list(line)
        for c in chars:
            if c == '.':
                num_periods += 1
    return num_periods

############################################################
## 'ITEM' detector
############################################################
def item_detector(text):
    score = 0
    weight_score = 0.75
    for line in text:
        if re.match(r'^\t*\s*\-', line):
            score += 1
        if re.match(r'^\s*[0-9]\.\s|^\s*\([0-9]\)\s', line):
            score += 10
    return 1 if (score / len(text)) >= weight_score else 0

############################################################
## Checks whether every line is indented by same number
############################################################
def same_indent(text):
    num_spaces = []
    equal_spaces = 1

    for line in text:
        num_spaces.append(indentation_length(line))
    for i in range(len(num_spaces) - 1):
        if num_spaces[i] != num_spaces[i+1]:
            equal_spaces = 0
    return equal_spaces

############################################################
## Returns indentation length
############################################################
def indentation_length(line):
    matchObj = re.match(r'(^\s+)', line)
    return len(matchObj.group(1)) if matchObj else 0

############################################################
## Ratio between number of whitespaces and non-whitespaces
############################################################
def white_space_ratio(text):
    num_chars = 0
    num_white_space = 0

    for line in text:
        chars = list(line)
        for c in chars:
            if c == ' ':
                num_white_space += 1
            num_chars += 1
    return num_white_space / num_chars

############################################################
## Ratio between number of lines that start with an
## alphabet and those that do not
############################################################
def begin_alphabet(text):
    cnt = 0
    regex = r'\s+[a-zA-Z].*'

    for line in text:
        if re.match(regex, line):
            cnt += 1
    return cnt / len(text)

############################################################
## 'ADDRESS' detector
############################################################
def address_detector(text):
    if quoted_detector(text) or item_detector(text):
        return 0

    for line in text:
        if re.match(r'.*[A-Z]{2}\s+[0-9]+.*', line) or re.match(r'.*[0-9]{3,5}-[0-9]{3,5}.*', line) or re.match(r'.*mail:.*@.*', line):
            return 1
    return 0

############################################################
## 'PTEXT' detector
############################################################
def text_detector(text):
    if quoted_detector(text) or item_detector(text):
        return 0
    num_text_line = 0
    line_started = 0
    white_space_cnt = 0

    for line in text:
        char_cnt = 0
        chars = list(line)
        for c in chars:
            if re.match(r'\s', c):
                white_space_cnt += 1
            else:
                white_space_cnt = 0
                line_started = 1
            if line_started:
                if white_space_cnt > 2:
                    return 0
                char_cnt += 1
        if char_cnt > 65:
            num_text_line += 1
    return num_text_line / len(text)

if __name__ == '__main__':
    main()