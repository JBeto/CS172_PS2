import math


def magnitude(vector):
    squared_sum = sum([v ** 2 for v in vector])
    return math.sqrt(squared_sum)


def normalize(vector):
    m = magnitude(vector)
    return [float(v) / m for v in vector]


# assuming query and document are normalized vectors
def cosine_similarity(query, document):
    dot_product = 0
    for q, d in zip(query, document):
        dot_product += float(q) * d
    return dot_product / (magnitude(query) * magnitude(document))
