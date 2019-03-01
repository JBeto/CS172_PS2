import re
from collections import namedtuple
from nltk.stem import PorterStemmer

# tags
_DOC_TAG = '<DOC>'
_TEXT_START_TAG = '<TEXT>'
_TEXT_END_TAG = '</TEXT>'
_DOC_NO_START_TAG = '<DOCNO>'
_DOC_NO_END_TAG = '</DOCNO>'

# stemmer
_ps = PorterStemmer()


Document = namedtuple('Document', ['doc_id', 'terms'])


def clean_terms(raw_text, stop_words):
    no_punctuation = [t for t in re.sub('[!#?,.:";-]', ' ', raw_text).split()]
    lower_case = [t.lower() for t in no_punctuation]
    raw_terms = [t for t in lower_case if t not in stop_words]
    stemmed_terms = [_ps.stem(t) for t in raw_terms]
    return stemmed_terms


def parse_documents(file_name, stop_words):
    with open(file_name, 'r') as f:
        raw_documents = f.read().split(_DOC_TAG)

    documents = []
    for doc in raw_documents:
        # document id
        doc_id = re.search('{} (.+?) {}'.format(_DOC_NO_START_TAG, _DOC_NO_END_TAG), doc)

        # raw terms from the <TEXT> tag
        raw_text = re.search('{}(.+?){}'.format(_TEXT_START_TAG, _TEXT_END_TAG), doc)
        terms = clean_terms(raw_text, stop_words)
        documents.append(Document(doc_id, terms))
    return documents
