import gensim
import os
import collections
import smart_open
import random
import csv
from scipy import spatial

def read_corpus(fname):
    with open(fname, mode='r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        for i, row in enumerate(csv_reader):
            if i%10000 == 0:
                print(i)
            yield row[3]

model = gensim.models.doc2vec.Doc2Vec.load('doc2vec_model.d2v')
print('model_loaded')
file = 'data/master_files/sorted_comments.csv'
gen = read_corpus(file)
first_comment = gen.__next__()
first_vector = model.infer_vector(gensim.utils.simple_preprocess(first_comment))
print (first_comment)

best_comment = gen.__next__()
best_sim = spatial.distance.cosine(first_vector, model.infer_vector(gensim.utils.simple_preprocess(best_comment)))

for comment in gen:
    sim = spatial.distance.cosine(first_vector, model.infer_vector(gensim.utils.simple_preprocess(comment)))
    if sim < best_sim:
        best_sim = sim
        best_comment = comment

print(best_comment)