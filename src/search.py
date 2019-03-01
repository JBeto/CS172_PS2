import math
import parser
from collections import defaultdict, Counter, namedtuple
from reinventing_the_wheel import cosine_similarity


Posting = namedtuple('Posting', ['doc_id', 'frequency'])

Scoring = namedtuple('Scoring', ['tf_idf', 'doc_id'])


def _get_stop_words(stop_words_file):
    with open(stop_words_file) as f:
        return set([x for x in f.read().split()])


def _terms_to_frequency(terms):
    return dict(Counter(terms))


class DocIndex:
    def __init__(self, documents_file, stop_words_file):
        self.posting_index = defaultdict(list)
        self.doc_id_index = {}
        self.stop_words = _get_stop_words(stop_words_file)

        documents = parser.parse_documents(documents_file, self.stop_words)
        for d in documents:
            terms = _terms_to_frequency(d.terms)
            # doc_id_index
            self.doc_id_index[d.doc_id] = len(terms)
            # posting_index
            for term, freq in terms.items():
                self.posting_index[term].append(Posting(d.doc_id, freq))
        
    def total_documents(self):
        return len(self.doc_id_index)

    def documents_appeared(self, term):
        return len(self.posting_index[term])

    def raw_term_frequency(self, doc_id, term):
        for p in self.posting_index[term]:
            if p.doc_id == doc_id:
                return p.freq
        return 0

    def query(self, query, n):
        query_terms = parser.clean_terms(query, self.stop_words)

        # dictionary that maps doc_id -> document vector
        document_vectors = defaultdict(lambda: [0] * len(query_terms))

        for i in range(len(query_terms)):
            term = query_terms[i]
            idf = 1 + math.log(float(self.total_documents()) / self.documents_appeared(term))
            for doc_id in self.doc_id_index:
                tf = float(self.raw_term_frequency(doc_id, term)) / self.doc_id_index[doc_id]
                document_vectors[doc_id][i] = tf * idf

        return sorted([(cosine_similarity(v, query), d) for d, v in document_vectors])[:n]
