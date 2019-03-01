from search import DocIndex
from pathlib import Path


def run_queries():
    n = 100
    doc_index = DocIndex(Path('data') / 'ap89_collection', 'stoplist.txt')
    with open(Path('data') / 'query_list.txt', 'r') as f:
        for query_line in f.readlines():
            period_index = query_line.find('.')
            query_number = query_line[:period_index]
            query = query_line[period_index+1:]
            results = doc_index.query(query, n)

            # output
            for i in range(n):
                print('{query_no} Q0 {docno} {rank} {score} Exp'.format(
                    query_no=query_number,
                    docno=results[i][1],
                    rank=i+1,
                    score=results[i][0]))


if __name__ == '__main__':
    run_queries()
