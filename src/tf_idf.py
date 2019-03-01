import os
import re
import math
from collections import defaultdict, Counter
from pathlib import Path
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

def get_tf(term_frequency, total_terms):
    return term_frequency / float(total_terms)


def get_idf(num_docs, docs_appeared):
    return math.log(num_docs / float(docs_appeared), 2.0)


def get_tf_idf(tf, idf):
    return tf * idf


class Posting:
    __slots__ = ['doc_id', 'frequency']

    def __init__(self, doc_id, frequency):
        self.doc_id = doc_id
        self.frequency = frequency


def _get_stop_words(words_file):
    with open(words_file) as f:
        return set([x.lower() for x in re.sub('[!#?,.:";-]', '', f.read()).split()])


def _get_terms(words_file, stop_words):
    with open(words_file) as f:
        # filter for stop words and punctuation
        terms = [x.lower() for x in re.sub('[!#?,.:";-]', '', f.read()).split() if x not in stop_words]
        # apply stemming
        terms = [ps.stem(t) for t in terms]
        return dict(Counter(terms))


class DocIndex:
    def __init__(self, input_directory, stop_words_file):
        self.posting_index = defaultdict(list)
        self.doc_id_index = {}

        stop_words = _get_stop_words(stop_words_file)

        doc_id = 1
        for f in sorted([f for f in os.listdir(input_directory)]):
            terms = _get_terms(str(Path(input_directory) / f), stop_words)
            # doc_id_index
            self.doc_id_index[doc_id] = len(terms)
            # posting_index
            for term, freq in terms.items():
                self.posting_index[term].append(Posting(doc_id, freq))
            doc_id += 1
        
    def get_total_documents(self):
        return len(self.doc_id_index)
