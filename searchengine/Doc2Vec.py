import gensim
import os
import collections
import smart_open
import random
import csv

def read_lee(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        print('x')
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def read_lee2(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

def read_corpus(fname, tokens_only=False):
    with open(fname, mode='r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        next(csv_reader, None)
        for i, row in enumerate(csv_reader):
            if i%1000000 == 0:
                print(i)
                if i==50000000:
                    return
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[3]), [i])

train_file = 'data/master_files/sorted_comments.csv'
#train_file = 'lee_background.cor'

#train_corpus = list(read_lee(train_file))
print("HERE0")
model = gensim.models.doc2vec.Doc2Vec(size=100, min_count=3, iter=5, workers=5)
print("HERE1")
model.build_vocab(read_corpus(train_file))
#model.build_vocab(train_corpus)

print("HERE2")
for i in range(model.iter):
    model.train(read_corpus(train_file), total_examples=model.corpus_count, epochs=1)
    model.save('model' + str(i) + '.d2v')
#model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)

print("DONE")

print(model.infer_vector(['only', 'you', 'can', 'prevent', 'forest' 'fires']))

model.save('model.d2v')