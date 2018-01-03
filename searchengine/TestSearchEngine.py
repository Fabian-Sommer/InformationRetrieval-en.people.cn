#!/usr/bin/python3

import unittest
import filecmp

from IndexCreator import *
from SearchEngine import *

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        # self.maxDiff = None
        self.test_directory = 'data/test'
        self.index_creator = IndexCreator(self.test_directory)
        self.index_creator.create_index()
        self.search_engine = SearchEngine()
        self.search_engine.load_index(self.test_directory)

    def tearDown(self):
        for path in [
            'index.csv',
            'collection_term_count.pickle',
            'comment_term_count_dict.pickle',
            'compressed_index',
            'compressed_seek_list.pickle',
            'seek_list.pickle',
            'symbol_encoding_pairs.pickle',
            'huffman_tree.pickle'
        ]:
            if os.path.isfile(f'{self.test_directory}/{path}'):
                os.remove(f'{self.test_directory}/{path}')

        self.search_engine.index_file.close()
        self.search_engine.comment_file.close()

    def test_dirichlet_smoothed_score(self):
        # TODO
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
