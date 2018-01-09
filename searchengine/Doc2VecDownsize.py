import gensim
import os
import collections
import smart_open
import random
import csv


model = gensim.models.doc2vec.Doc2Vec.load('model.d2v')

model.delete_temporary_training_data(keep_doctags_vectors=False)
model.save('doc2vec_model.d2v')