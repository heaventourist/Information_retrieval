import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, List, NamedTuple

import numpy as np
from numpy.linalg import norm
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import math

### File IO and processing

class Document(NamedTuple):
    doc_id: int
    author: List[str]
    title: List[str]
    keyword: List[str]
    abstract: List[str]

    def sections(self):
        return [self.author, self.title, self.keyword, self.abstract]

    def __repr__(self):
        return (f"doc_id: {self.doc_id}\n" +
            f"  author: {self.author}\n" +
            f"  title: {self.title}\n" +
            f"  keyword: {self.keyword}\n" +
            f"  abstract: {self.abstract}")


def read_stopwords(file):
    with open(file) as f:
        return set([x.strip() for x in f.readlines()])

stopwords = read_stopwords('common_words')

stemmer = SnowballStemmer('english')

def read_rels(file):
    '''
    Reads the file of related documents and returns a dictionary of query id -> list of related documents
    '''
    rels = {}
    with open(file) as f:
        for line in f:
            qid, rel = line.strip().split()
            qid = int(qid)
            rel = int(rel)
            if qid not in rels:
                rels[qid] = []
            rels[qid].append(rel)
    return rels

def read_docs(file):
    '''
    Reads the corpus into a list of Documents
    '''
    docs = [defaultdict(list)]  # empty 0 index
    category = ''
    with open(file) as f:
        i = 0
        for line in f:
            line = line.strip()
            if line.startswith('.I'):
                i = int(line[3:])
                docs.append(defaultdict(list))
            elif re.match(r'\.\w', line):
                category = line[1]
            elif line != '':
                for word in word_tokenize(line):
                    docs[i][category].append(word.lower())

    return [Document(i + 1, d['A'], d['T'], d['K'], d['W'])
        for i, d in enumerate(docs[1:])]


DOCS = read_docs('cacm.raw')


def stem_doc(doc: Document):
    return Document(doc.doc_id, *[[stemmer.stem(word) for word in sec]
        for sec in doc.sections()])

def stem_docs(docs: List[Document]):
    return [stem_doc(doc) for doc in docs]

def remove_stopwords_doc(doc: Document):
    return Document(doc.doc_id, *[[word for word in sec if word not in stopwords]
        for sec in doc.sections()])

def remove_stopwords(docs: List[Document]):
    return [remove_stopwords_doc(doc) for doc in docs]



### Term-Document Matrix

class TermWeights(NamedTuple):
    author: float
    title: float
    keyword: float
    abstract: float

def compute_doc_freqs(docs: List[Document]):
    '''
    Computes document frequency, i.e. how many documents contain a specific word
    '''
    freq = Counter()
    for doc in docs:
        words = set()
        for sec in doc.sections():
            for word in sec:
                words.add(word)
        for word in words:
            freq[word] += 1
    return freq

def compute_tf(doc: Document, doc_freqs: Dict[str, int], weights: list):
    vec = defaultdict(float)
    for word in doc.author:
        vec[word] += weights.author
    for word in doc.keyword:
        vec[word] += weights.keyword
    for word in doc.title:
        vec[word] += weights.title
    for word in doc.abstract:
        vec[word] += weights.abstract
    return dict(vec)  # convert back to a regular dict

def compute_tfidf(doc, doc_freqs, weights):
    tf = compute_tf(doc, doc_freqs, weights)
    N = len(DOCS)
    return {key: tf[key] * math.log(N / doc_freqs[key]) if doc_freqs[key] > 0 else 0 for key in tf.keys()}  # TODO: implement

def compute_boolean(doc, doc_freqs, weights):
    tf = compute_tf(doc, doc_freqs, weights)
    return {key: 1 for key in tf.keys()}  # TODO: implement



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
    # num = dictdot(x, y)
    # if num == 0:
    #     return 0
    #
    # try:
    #     return num / (sum(list(x.values())) + sum(list(y.values())) - num)  # TODO: implement
    # except ZeroDivisionError:
    #     return 1
    keys = list(x.keys()) if len(x) < len(y) else list(y.keys())
    try:
        return sum([min(x.get(key,0), y.get(key,0)) for key in keys])/sum([max(x.get(key,0), y.get(key,0)) for key in keys])
    except ZeroDivisionError:
        return 1

def overlap_sim(x, y):
    num = dictdot(x, y)
    if num == 0:
        return 0
    return num / min(sum(list(x.values())), sum(list(y.values())))  # TODO: implement


### Precision/Recall

def interpolate(x1, y1, x2, y2, x):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m * x + b


def precision_at(recall: float, results: List[int], relevant: List[int]) -> float:
    '''
    This function should compute the precision at the specified recall level.
    If the recall level is in between two points, you should do a linear interpolation
    between the two closest points. For example, if you have 4 results
    (recall 0.25, 0.5, 0.75, and 1.0), and you need to compute recall @ 0.6, then do something like

    interpolate(0.5, prec @ 0.5, 0.75, prec @ 0.75, 0.6)

    Note that there is implicitly a point (recall=0, precision=1).

    `results` is a sorted list of document ids
    `relevant` is a list of relevant documents
    '''
    tmp_results = list()
    tmp_results.append((0, 1))

    tmp_set = set(relevant)
    cnt = 0
    for index, doc_id in enumerate(results):
        if doc_id in tmp_set:
            cnt += 1
            m_precision = cnt / (index+1)
            m_recall = cnt / len(relevant)
            tmp_results.append((m_recall, m_precision))
    tmp_results.sort(key=lambda x: x[0])
    #print(tmp_results)
    prev = None
    for item in tmp_results:
        if item[0] == recall:
            return item[1]
        elif item[0] < recall:
            prev = item
        else:
            return interpolate(prev[0], prev[1], item[0], item[1], recall)

def mean_precision1(results, relevant):
    return (precision_at(0.25, results, relevant) +
        precision_at(0.5, results, relevant) +
        precision_at(0.75, results, relevant)) / 3

def mean_precision2(results, relevant):
    return sum(list(precision_at(i/10, results, relevant) for i in range(1, 11)))/10  # TODO: implement

def norm_recall(results, relevant):
    N = len(DOCS)
    Rel = len(relevant)
    return 1 - (sum(list(results.index(i)+1 if i in results else 0 for i in relevant)) -
        sum(list(i for i in range(1, Rel+1)))) / (Rel*(N - Rel))  # TODO: implement

def norm_precision(results, relevant):
    N = len(DOCS)
    Rel = len(relevant)
    return 1 - (sum(list(math.log(results.index(i)+1) if i in results else 0 for i in relevant)) -
        sum(list(math.log(i) for i in range(1, Rel+1)))) / (N*math.log(N) - (N -Rel)*math.log(N-Rel) - Rel*math.log(Rel))  # TODO: implement


### Extensions

# TODO: put any extensions here

def get_total_words():
    docs = read_docs('cacm.raw')
    total_words = set()
    for doc in docs:
        for sec in doc.sections():
            for word in sec:
                total_words.add(word)
    total_words = list(total_words)
    total_words.sort()
    return total_words


def meanX(dataX):
    return np.mean(dataX,axis=0)

def svd(XMat, k):
    average = meanX(XMat)
    m, n = np.shape(XMat)
    avgs = np.tile(average, (m, 1))
    data_adjust = XMat - avgs
    covX = np.cov(data_adjust.T)
    featValue, featVec= np.linalg.eig(covX)
    index = np.argsort(-featValue)
    if k > n:
        print("k must lower than feature number")
        return
    else:
        selectVec = np.matrix(featVec.T[index[:k]])
        finalData = data_adjust * selectVec.T
        reconData = (finalData * selectVec) + average
    return reconData, index[:k]


total_words = get_total_words()

def convert(docVecs, queryVecs):
    res = [list(map(lambda x: vec[x] if x in vec else 0, total_words)) for vec in docVecs]
    docSize = len(res)
    res += [list(map(lambda x: vec[x] if x in vec else 0, total_words)) for vec in queryVecs]
    tmp_matrix = np.asmatrix(res)
    final_matrix, indices = svd(tmp_matrix, int(len(total_words)*0.9))
    docMatrix = final_matrix[:docSize]
    queryMatrix = final_matrix[docSize:]
    newDocVecs = [dict() for i in range(docMatrix.shape[0])]
    newQueryVecs = [dict() for i in range(queryMatrix.shape[0])]

    reduced_total_words = list(filter(lambda x: total_words.index(x) in indices, total_words))

    for j, word in enumerate(reduced_total_words):
        for i, d_vec in enumerate(newDocVecs):
            if docMatrix[i, j] != 0:
                d_vec[word] = docMatrix[i, j]

    for j, word in enumerate(reduced_total_words):
        for i, d_vec in enumerate(newQueryVecs):
            if queryMatrix[i, j] != 0:
                d_vec[word] = queryMatrix[i, j]
    return newDocVecs, newQueryVecs


def SVDExperiment():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        #'tf': compute_tf,
        'tfidf': compute_tfidf
        #'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim
        #'jaccard': jaccard_sim,
        #'dice': dice_sim,
        #'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
         TermWeights(author=3, title=3, keyword=4, abstract=1),
         TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    # term_funcs = {
    #     'tf': compute_tf,
    #     'tfidf': compute_tfidf,
    #     'boolean': compute_boolean
    # }
    #
    # sim_funcs = {
    #     'cosine': cosine_sim,
    #     'jaccard': jaccard_sim,
    #     'dice': dice_sim,
    #     'overlap': overlap_sim
    # }
    #
    # permutations = [
    #     term_funcs,
    #     [False, True],  # stem
    #     [False, True],  # remove stopwords
    #     sim_funcs,
    #     [TermWeights(author=1, title=1, keyword=1, abstract=1),
    #      TermWeights(author=3, title=3, keyword=4, abstract=1),
    #      TermWeights(author=1, title=1, keyword=1, abstract=4)]
    # ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]
        query_vectors = [term_funcs[term](query, doc_freqs, term_weights) for query in processed_queries]
        query_ids = [query.doc_id for query in processed_queries]

        doc_vectors, query_vectors = convert(doc_vectors, query_vectors)
        metrics = []

        for index, query_vec in enumerate(query_vectors):
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query_ids[index]]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])

        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

        #return  # TODO: just for testing; remove this when printing the full table



### Search

def experiment():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')

    term_funcs = {
        'tf': compute_tf,
        'tfidf': compute_tfidf,
        'boolean': compute_boolean
    }

    sim_funcs = {
        'cosine': cosine_sim,
        'jaccard': jaccard_sim,
        'dice': dice_sim,
        'overlap': overlap_sim
    }

    permutations = [
        term_funcs,
        [False, True],  # stem
        [False, True],  # remove stopwords
        sim_funcs,
        [TermWeights(author=1, title=1, keyword=1, abstract=1),
            TermWeights(author=3, title=3, keyword=4, abstract=1),
            TermWeights(author=1, title=1, keyword=1, abstract=4)]
    ]

    print('term', 'stem', 'removestop', 'sim', 'termweights', 'p_0.25', 'p_0.5', 'p_0.75', 'p_1.0', 'p_mean1', 'p_mean2', 'r_norm', 'p_norm', sep='\t')

    # This loop goes through all permutations. You might want to test with specific permutations first
    for term, stem, removestop, sim, term_weights in itertools.product(*permutations):
        processed_docs, processed_queries = process_docs_and_queries(docs, queries, stem, removestop, stopwords)
        doc_freqs = compute_doc_freqs(processed_docs)
        doc_vectors = [term_funcs[term](doc, doc_freqs, term_weights) for doc in processed_docs]

        metrics = []

        for query in processed_queries:
            query_vec = term_funcs[term](query, doc_freqs, term_weights)
            results = search(doc_vectors, query_vec, sim_funcs[sim])
            # results = search_debug(processed_docs, query, rels[query.doc_id], doc_vectors, query_vec, sim_funcs[sim])
            rel = rels[query.doc_id]

            metrics.append([
                precision_at(0.25, results, rel),
                precision_at(0.5, results, rel),
                precision_at(0.75, results, rel),
                precision_at(1.0, results, rel),
                mean_precision1(results, rel),
                mean_precision2(results, rel),
                norm_recall(results, rel),
                norm_precision(results, rel)
            ])
        #print(metrics)
        averages = [f'{np.mean([metric[i] for metric in metrics]):.4f}'
            for i in range(len(metrics[0]))]
        print(term, stem, removestop, sim, ','.join(map(str, term_weights)), *averages, sep='\t')

        #return  # TODO: just for testing; remove this when printing the full table


def process_docs_and_queries(docs, queries, stem, removestop, stopwords):
    processed_docs = docs
    processed_queries = queries
    if removestop:
        processed_docs = remove_stopwords(processed_docs)
        processed_queries = remove_stopwords(processed_queries)
    if stem:
        processed_docs = stem_docs(processed_docs)
        processed_queries = stem_docs(processed_queries)
    return processed_docs, processed_queries


def search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]
    return results


def search_debug(docs, query, relevant, doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec))
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [x[0] for x in results_with_score]

    print('Query:', query)
    print('Relevant docs: ', relevant)
    print()
    for doc_id, score in results_with_score[:10]:
        print('Score:', score)
        print(docs[doc_id - 1])
        print()

def m_search(doc_vectors, query_vec, sim):
    results_with_score = [(doc_id + 1, sim(query_vec, doc_vec), doc_vec)
                    for doc_id, doc_vec in enumerate(doc_vectors)]
    results_with_score = sorted(results_with_score, key=lambda x: -x[1])
    results = [(x[0], x[2]) for x in results_with_score]
    return results

def listTop20_docs_stemmed():
    docs = read_docs('cacm.raw')
    queries = filter(lambda x: x.doc_id in [6, 9, 22], read_docs('query.raw'))
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')
    processed_docs, processed_queries = process_docs_and_queries(docs, queries, True, True, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    term_weights = TermWeights(author=3, title=3, keyword=4, abstract=1)
    doc_vectors = [compute_tfidf(doc, doc_freqs, term_weights) for doc in processed_docs]
    res = dict()

    for query in processed_queries:
        query_vec = compute_tfidf(query, doc_freqs, term_weights)
        results = search(doc_vectors, query_vec, cosine_sim)[:20]
        m_results = list()
        for m_id in results:
            m_results += list(filter(lambda x: x.doc_id == m_id, docs))
        rel = rels[query.doc_id]
        res[query.doc_id] = list(map(lambda x: [x.doc_id, ' '.join(x.title), x.doc_id in rel], m_results))

    print('query_id', 'term', 'stem', 'removestop', 'sim', 'termweights', 'rank', 'retrieved_docID','title', 'relevant?', sep='\t')

    for m_id in res.keys():
        for rank, item in enumerate(res[m_id]):
            print(m_id, 'tfidf', True, True, 'cosine', ','.join(map(str, term_weights)), rank+1, *item, sep='\t')


def listTop20_docs_unstemmed():
    docs = read_docs('cacm.raw')
    queries = filter(lambda x: x.doc_id in [6, 9, 22], read_docs('query.raw'))
    rels = read_rels('query.rels')
    stopwords = read_stopwords('common_words')
    processed_docs, processed_queries = process_docs_and_queries(docs, queries, False, True, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    term_weights = TermWeights(author=3, title=3, keyword=4, abstract=1)
    doc_vectors = [compute_tfidf(doc, doc_freqs, term_weights) for doc in processed_docs]
    res = dict()

    for query in processed_queries:
        query_vec = compute_tfidf(query, doc_freqs, term_weights)
        results = search(doc_vectors, query_vec, cosine_sim)[:20]
        m_results = list()
        for m_id in results:
            m_results += list(filter(lambda x: x.doc_id == m_id, docs))
        rel = rels[query.doc_id]
        res[query.doc_id] = list(map(lambda x: [x.doc_id, ' '.join(x.title), x.doc_id in rel], m_results))

    print('query_id', 'term', 'stem', 'removestop', 'sim', 'termweights', 'rank','retrieved_docID','title', 'relevant?', sep='\t')

    for m_id in res.keys():
        for rank, item in enumerate(res[m_id]):
            print(m_id, 'tfidf', False, True, 'cosine', ','.join(map(str, term_weights)), rank+1, *item, sep='\t')


def listTop10_term_weights():
    docs = read_docs('cacm.raw')
    queries = read_docs('query.raw')
    stopwords = read_stopwords('common_words')
    processed_docs, processed_queries = process_docs_and_queries(docs, queries, True, True, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    term_weights = TermWeights(author=3, title=3, keyword=4, abstract=1)
    doc_vectors = [compute_tfidf(doc, doc_freqs, term_weights) for doc in processed_docs]
    record = dict()

    for query in processed_queries:
        query_vec = compute_tfidf(query, doc_freqs, term_weights)
        results = m_search(doc_vectors, query_vec, cosine_sim)[:10]
        record[query.doc_id] = {'query_vector': query_vec, 'doc_vectors': list()}
        for d_id, d_vec in results:
            record[query.doc_id]['doc_vectors'].append([d_id, d_vec])


    print('query_id', 'term', 'stem', 'removestop', 'sim', 'termweights', 'rank', 'doc_id', 'terms', 'weight', 'term_from', sep='\t')

    for q_id in record.keys():
        for item in record[q_id]['query_vector'].keys():
            if record[q_id]['query_vector'][item] > 0:
                print(q_id, 'tfidf', True, True, 'cosine', ','.join(map(str, term_weights)), 'NA', 'NA', item, record[q_id]['query_vector'][item], 'query', sep='\t')

        for rank, item1 in enumerate(record[q_id]['doc_vectors']):
            for item2 in item1[1].keys():
                if item1[1][item2] > 0:
                    print(q_id, 'tfidf', True, True, 'cosine', ','.join(map(str, term_weights)), rank+1, item1[0], item2, item1[1][item2], 'doc', sep='\t')

def listTop20_similarTo_239_1236_2740():
    docs = read_docs('cacm.raw')
    queries = list(filter(lambda x: x.doc_id in [239, 1236, 2740], docs))
    stopwords = read_stopwords('common_words')
    processed_docs, processed_queries = process_docs_and_queries(docs, queries, True, True, stopwords)
    doc_freqs = compute_doc_freqs(processed_docs)
    processed_docs = list(filter(lambda x: x.doc_id not in [239, 1236, 2740], processed_docs))
    term_weights = TermWeights(author=3, title=3, keyword=4, abstract=1)
    doc_vectors = [compute_tfidf(doc, doc_freqs, term_weights) for doc in processed_docs]
    record = dict()

    for query in processed_queries:
        query_vec = compute_tfidf(query, doc_freqs, term_weights)
        results = search(doc_vectors, query_vec, cosine_sim)[:20]
        record[query.doc_id] = list()
        for d_id in results:
            m_doc = list(filter(lambda x: x.doc_id == d_id, docs))[0]
            record[query.doc_id].append([d_id, ' '.join(m_doc.title)])

    print('query_id', 'term', 'stem', 'removestop', 'sim', 'termweights', 'rank', 'doc_id', 'title', sep='\t')
    for q_id in record.keys():
        for rank, item in enumerate(record[q_id]):
            print(q_id, 'tfidf', True, True, 'cosine', ','.join(map(str, term_weights)), rank+1, *item, sep='\t')

if __name__ == '__main__':
    # I have given the results for running each of these functions, SVD extension is extremely slow, be patient!
    
    # regular experiment
    experiment()

    # # List the top 20 retrieved documents for eries 6, 9 and 22 by their number, title and similarity
    # # measure, with the “relevant” documents starred.
    # listTop20_docs_stemmed()
    # listTop20_docs_unstemmed()
    #
    # # For the top 10 retrieved documents, show the terms on which the retrieval was based (those with
    # # non-zero weights for both query and retrieved document) along with these weights.
    # listTop10_term_weights()
    #
    # # List the top 20 documents that are most similar to Documents 239, 1236 and 2740, giving number,
    # # title and similarity measure
    # listTop20_similarTo_239_1236_2740()
    #
    # # SVD extension
    # # Attention: SVD processing is extremely slow, a single round might take more than 10 minutes.
    # SVDExperiment()