from tf_idf import DocIndex, get_tf, get_idf, get_tf_idf


def test():
    doc_index = DocIndex('data', 'stoplist.txt')
    term = input('Enter a term: ')
    while term != 'QUIT':
        if term not in doc_index.posting_index:
            print("Term does not exists in index.")
        else:
            docs_appeared = len(doc_index.posting_index[term])
            for posting in doc_index.posting_index[term]:
                tf = get_tf(posting.frequency, doc_index.doc_id_index[posting.doc_id])
                idf = get_idf(doc_index.get_total_documents(), docs_appeared)
                tf_idf = get_tf_idf(tf, idf)

                print(tf, end=',')
                print(idf, end=',')
                print(tf_idf)
        term = input('Enter a term: ')


if __name__ == '__main__':
    test()
