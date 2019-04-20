from typing import Dict, List, NamedTuple
from collections import Counter, defaultdict
import math
import re
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import random


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')
stemmer = SnowballStemmer('english')
class Document(NamedTuple):
    doc_id: int
    text: List[str]

    def sections(self):
        return [self.text]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  text: {self.text}")


def read_docs(file, mode='bow'):
    docs = list()
    labels = list()
    if re.search('smsspam', file) or re.search('sent_imdb', file) or re.search('sent_yelp', file) or re.search('health', file):
        with open(file) as f:
            for line in f:
                try:
                    doc_id, label, text = line.strip().split('\t')
                    text = [word for word in word_tokenize(text)]
                    labels.append(int(label))
                    docs.append(text)
                except:
                    continue
    elif mode == 'bow':
        with open(file) as f:
            for line in f:
                try:
                    doc_id, label, text = line.strip().split('\t')
                    text = [word for word in text.strip().split()]
                    if len(list(filter(lambda x: re.match(r'^\.X-\S+$', x), text))) == 0:
                        continue
                    labels.append(int(label))
                    docs.append(text)
                except:
                    continue
    elif mode == 'bigram':
        with open(file) as f:
            for line in f:
                try:
                    doc_id, label, text = line.strip().split('\t')
                    text = [word for word in text.strip().split()]
                    target_word = list(filter(lambda x: re.match(r'^\.X-\S+$', x), text))[0]
                    labels.append(int(label))
                    bigram_text = list()
                    for i, word in enumerate(text):
                        if i == text.index(target_word)-1:
                            bigram_text.append(text[i] + ' ' + text[i+1])
                        elif i == text.index(target_word):
                            continue
                        elif i == text.index(target_word)+1:
                            bigram_text.append(text[i-1] + ' ' + text[i])
                        else:
                            bigram_text.append(word)
                    docs.append(bigram_text)
                except:
                    continue
    return [Document(i+1, text) for i, text in enumerate(docs)], labels


def get_profile(vecs, label, labels):
    profile = defaultdict(float)
    vecs = list(filter(lambda x: labels[vecs.index(x)] == label, vecs))
    for vec in vecs:
        for key in vec.keys():
            profile[key] += vec[key]
    for key in profile.keys():
        profile[key] /= len(vecs)
    return dict(profile)

def get_sum(vecs, label, labels):
    profile = defaultdict(float)
    vecs = list(filter(lambda x: labels[vecs.index(x)] == label, vecs))
    for vec in vecs:
        for key in vec.keys():
            profile[key] += vec[key]
    return dict(profile)


def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        for word in doc.text:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, weights: List[float]):
    vec = defaultdict(float)
    for i, word in enumerate(doc.text):
        vec[word] += weights[i]
    return dict(vec)


def compute_tfidf(doc, doc_freqs, weights, docs_size):
    tf = compute_tf(doc, weights)
    return {key: tf[key] * math.log(docs_size / doc_freqs[key]) if doc_freqs[key] > 0 else 0 for key in tf.keys()}


### Vector Similarity

def dictdot(x: Dict[str, float], y: Dict[str, float]):
    '''
    Computes the dot product of vectors x and y, represented as sparse dictionaries.
    '''
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    return sum(x.get(key, 0) * y.get(key, 0) for key in keys)

def cosine_sim(x, y):
    '''
    Computes the cosine similarity between two sparse term vectors represented as dictionaries.
    '''
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / (norm(list(x.values())) * norm(list(y.values())))


def dice_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0

    return 2*num / (sum(list(x.values())) + sum(list(y.values())))  # TODO: implement


def jaccard_sim(x, y):
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    try:
        return sum([min(x.get(key,0), y.get(key,0)) for key in keys])/sum([max(x.get(key,0), y.get(key,0)) for key in keys])
    except ZeroDivisionError:
        return 1

def overlap_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / min(sum(list(x.values())), sum(list(y.values())))

def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) if not re.match(r'^\.X-\S+$', word) else word for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word.lower() not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]

def exponential_weighting(docs, mode='bow'):
    weights = list()
    if mode == 'bow':
        for doc in docs:
            target_word = list(filter(lambda x: re.match(r'^\.X-\S+$', x), doc.text))[0]
            tmp = [1/abs(doc.text.index(target_word)-doc.text.index(key)) if key != target_word else 0 for key in doc.text]
            weights.append(tmp)
    elif mode == 'bigram':
        for doc in docs:
            target_word = list(filter(lambda x: re.match(r'^\S*\s\.X-\S+$', x), doc.text))[0]
            tmp = [1/(doc.text.index(target_word)-i+1) if i <= doc.text.index(target_word) else 1/(i-doc.text.index(target_word)) for i in range(len(doc.text))]
            weights.append(tmp)
    return weights


def stepped_weighting(docs, mode='bow'):
    weights = list()
    if mode == 'bow':
        for doc in docs:
            target_word = list(filter(lambda x: re.match(r'^\.X-\S+$', x), doc.text))[0]
            tmp = [1 if abs(doc.text.index(target_word)-doc.text.index(key)) > 3 else [0,6,3,3][abs(doc.text.index(target_word)-doc.text.index(key))] for key in doc.text]
            weights.append(tmp)
    elif mode == 'bigram':
        for doc in docs:
            target_word = list(filter(lambda x: re.match(r'^\S*\s\.X-\S+$', x), doc.text))[0]
            index_target_word = doc.text.index(target_word)
            tmp = list()
            for i, word in enumerate(doc.text):
                if i<index_target_word-2 or i>index_target_word+3:
                    tmp.append(1)
                elif (index_target_word-i) in [1,2] or (i-index_target_word-1) in [1,2]:
                    tmp.append(3)
                else:
                    tmp.append(6)
            weights.append(tmp)
    return weights


def uniform_weighting(docs, mode='bow'):
    weights = list()
    for doc in docs:
        weights.append([1 for key in doc.text])
    return weights


def custom_weighting(docs, mode='bow'):
    weights = list()
    if mode == 'bow':
        for doc in docs:
            target_word = list(filter(lambda x: re.match(r'^\.X-\S+$', x), doc.text))[0]
            tmp = [1 if abs(doc.text.index(target_word)-doc.text.index(key)) > 10 else [0,100,90,80,70,60,50,40,30,20,10][abs(doc.text.index(target_word)-doc.text.index(key))] for key in doc.text]
            weights.append(tmp)
    elif mode == 'bigram':
        for doc in docs:
            target_word = list(filter(lambda x: re.match(r'^\S*\s\.X-\S+$', x), doc.text))[0]
            index_target_word = doc.text.index(target_word)
            tmp = list()
            for i, word in enumerate(doc.text):
                if i<index_target_word-9 or i>index_target_word+10:
                    tmp.append(1)
                elif i>=index_target_word-9 and i<=index_target_word:
                    tmp.append([100,90,80,70,60,50,40,30,20,10][index_target_word-i])
                else:
                    tmp.append([100,90,80,70,60,50,40,30,20,10][i-index_target_word-1])
            weights.append(tmp)
    return weights


def compute_llike(v_sum1, v_sum2, epsilon):
    llk = dict()
    for term in v_sum2.keys():
        if v_sum2[term] == 0:
            continue
        if v_sum1.get(term, -1) > 0:
            llk[term] = math.log(v_sum1[term]/v_sum2[term])
        else:
            llk[term] = math.log(epsilon/v_sum2[term])
    for term in v_sum1.keys():
        if term not in v_sum2 and v_sum1[term] != 0:
            llk[term] = math.log(v_sum1[term]/epsilon)
    return llk


def experiment():
    train_set = ['tank-train.tsv', 'plant-train.tsv', 'perplace-train.tsv', 'smsspam-train.tsv', 'sent_imdb-train.tsv', 'sent_yelp-train.tsv', 'health-train.tsv']
    dev_set = ['tank-dev.tsv', 'plant-dev.tsv', 'perplace-dev.tsv', 'smsspam-dev.tsv', 'sent_imdb-dev.tsv', 'sent_yelp-dev.tsv', 'health-dev.tsv']
    stemming = {'stemmed': True, 'unstemmed': False}
    weighting = {'#0-uniform': uniform_weighting, '#1-expndecay': exponential_weighting, '#2-stepped': stepped_weighting, '#3-custom': custom_weighting}
    modes = {'#1-bag-of-words': 'bow', '#2-adjacent-separate-LR': 'bigram'}
    combinations = [['unstemmed', '#0-uniform', '#1-bag-of-words'],
                    ['stemmed', '#1-expndecay', '#1-bag-of-words'],
                    ['unstemmed', '#1-expndecay', '#1-bag-of-words'],
                    ['unstemmed', '#1-expndecay', '#2-adjacent-separate-LR'],
                    ['unstemmed', '#2-stepped', '#1-bag-of-words'],
                    ['unstemmed', '#3-custom', '#1-bag-of-words']
                    ]

    print('stemming', 'Position Weighting', 'Local Collocation Modelling', 'tank', 'plant', 'pers/place', 'smsspam', 'sent_imdb', 'sent_yelp', 'health', 'comment', sep='\t')

    for remove_stp in [False, True]:
        for stem, weight, mode in combinations:
            percentages = list()
            for train, dev in zip(train_set, dev_set):
                if train in ['smsspam-train.tsv', 'sent_imdb-train.tsv', 'sent_yelp-train.tsv', 'health-train.tsv']:
                    docs, labels = read_docs(train)
                    docs = remove_stopwords(docs) if remove_stp else docs
                    docs = stem_docs(docs) if stemming[stem] else docs
                    doc_freqs = compute_doc_freqs(docs)
                    weights = uniform_weighting(docs)
                    vecs = [compute_tfidf(doc, doc_freqs, weights[i], len(docs)) for i, doc in enumerate(docs)]
                    v_profile1 = get_profile(vecs, 1, labels)
                    v_profile2 = get_profile(vecs, 2, labels)

                    docs, labels = read_docs(dev)
                    docs = remove_stopwords(docs) if remove_stp else docs
                    docs = stem_docs(docs) if stemming[stem] else docs
                    doc_freqs = compute_doc_freqs(docs)
                    weights = uniform_weighting(docs)
                    vecs = [compute_tfidf(doc, doc_freqs, weights[i], len(docs)) for i, doc in enumerate(docs)]
                    sim1 = [cosine_sim(vec, v_profile1) for vec in vecs]
                    sim2 = [cosine_sim(vec, v_profile2) for vec in vecs]
                    correct_cnt = 0
                    incorrect_cnt = 0
                    for i in range(len(sim1)):
                        if sim1[i] > sim2[i]:
                            if labels[i] == 1:
                                correct_cnt += 1
                            else:
                                incorrect_cnt += 1
                        elif sim1[i] == sim2[i]:
                            correct_cnt += 1
                        else:
                            if labels[i] == 2:
                                correct_cnt += 1
                            else:
                                incorrect_cnt += 1
                    percentage = correct_cnt / (correct_cnt + incorrect_cnt)
                    percentages.append(round(percentage,4))
                else:
                    docs, labels = read_docs(train, mode=modes[mode])
                    docs = remove_stopwords(docs) if remove_stp else docs
                    docs = stem_docs(docs) if stemming[stem] else docs
                    doc_freqs = compute_doc_freqs(docs)
                    weights = weighting[weight](docs, modes[mode])
                    vecs = [compute_tfidf(doc, doc_freqs, weights[i], len(docs)) for i, doc in enumerate(docs)]
                    v_profile1 = get_profile(vecs, 1, labels)
                    v_profile2 = get_profile(vecs, 2, labels)

                    docs, labels = read_docs(dev, mode=modes[mode])
                    docs = remove_stopwords(docs) if remove_stp else docs
                    docs = stem_docs(docs) if stemming[stem] else docs
                    doc_freqs = compute_doc_freqs(docs)
                    weights = weighting[weight](docs, modes[mode])
                    vecs = [compute_tfidf(doc, doc_freqs, weights[i], len(docs)) for i, doc in enumerate(docs)]
                    sim1 = [cosine_sim(vec, v_profile1) for vec in vecs]
                    sim2 = [cosine_sim(vec, v_profile2) for vec in vecs]
                    correct_cnt = 0
                    incorrect_cnt = 0
                    for i in range(len(sim1)):
                        if sim1[i] > sim2[i]:
                            if labels[i] == 1:
                                correct_cnt += 1
                            else:
                                incorrect_cnt += 1
                        elif sim1[i] == sim2[i]:
                            correct_cnt += 1
                        else:
                            if labels[i] == 2:
                                correct_cnt += 1
                            else:
                                incorrect_cnt += 1
                    percentage = correct_cnt / (correct_cnt + incorrect_cnt)
                    percentages.append(round(percentage,4))
            print(stem, weight, mode, *percentages, 'stopwords removed' if remove_stp else 'no stopwords removed', sep='\t')
    bayes()


def bayes():
    train_set = ['tank-train.tsv', 'plant-train.tsv', 'perplace-train.tsv', 'smsspam-train.tsv', 'sent_imdb-train.tsv', 'sent_yelp-train.tsv', 'health-train.tsv']
    dev_set = ['tank-dev.tsv', 'plant-dev.tsv', 'perplace-dev.tsv', 'smsspam-dev.tsv', 'sent_imdb-dev.tsv', 'sent_yelp-dev.tsv', 'health-dev.tsv']
    percentages = list()
    for train, dev in zip(train_set, dev_set):
        docs, labels = read_docs(train)
        docs = remove_stopwords(docs)
        docs = stem_docs(docs)
        weights = exponential_weighting(docs) if train not in ['smsspam-train.tsv', 'sent_imdb-train.tsv', 'sent_yelp-train.tsv', 'health-train.tsv'] else uniform_weighting(docs)
        vecs = [compute_tf(doc, weights[i]) for i, doc in enumerate(docs)]
        v_sum1 = get_sum(vecs, 1, labels)
        v_sum2 = get_sum(vecs, 2, labels)
        llk = compute_llike(v_sum1, v_sum2, 0.2)

        docs, labels = read_docs(dev)
        docs = remove_stopwords(docs)
        docs = stem_docs(docs)
        weights = exponential_weighting(docs) if train not in ['smsspam-train.tsv', 'sent_imdb-train.tsv', 'sent_yelp-train.tsv', 'health-train.tsv'] else uniform_weighting(docs)
        vecs = [compute_tf(doc, weights[i]) for i, doc in enumerate(docs)]

        predicted = list()
        for i, vec in enumerate(vecs):
            sumofLL = 0
            for term in vec.keys():
                sumofLL += llk.get(term, 0)*vec[term]
            if sumofLL > 0:
                predicted.append(1)
            elif sumofLL < 0:
                predicted.append(2)
            else:
                predicted.append(random.choice([1,2]))

        correct_cnt = 0
        incorrect_cnt = 0
        for i,j in zip(labels, predicted):
            if i==j:
                correct_cnt += 1
            else:
                incorrect_cnt += 1
        percentage = correct_cnt / (correct_cnt + incorrect_cnt)
        percentages.append(round(percentage,4))
    print('stemmed', '#1-expndecay', '#1-bag-of-words', *percentages, 'bayes', sep='\t')



if __name__ == '__main__':
    experiment()
