#!/usr/bin/python3

import unittest

from IndexCreator import *

expected_index = r'''"!":1:547,1
"$":1:547,3
"%":3:547,4,5,7
"&":3:547,6,8,10
"''":1:547,23
"(":2:547,12,18
")":1:547,19
".":3:0,5:418,1:547,21
"...":1:0,2
".profound":1:0,3
"/":2:547,9,11
"/|":1:547,13
">":1:547,14
"?":1:161,1
"againsta§":1:547,0
"event":1:0,1
"goodinterna":1:289,0
"int":1:418,2
"intern":2:0,6:161,2
"interna":1:547,22
"lesson":1:0,4
"rise":1:418,0
"tragic":1:0,0
"xi":1:161,0
"~°¿á\\\":1:547,15
"§":1:547,2
"«":1:547,16
"»":1:547,17
"–":1:547,20
'''

class TestIndexCreation(unittest.TestCase):
    def setUp(self):
        # self.maxDiff = None
        self.test_directory = 'data/test'
        self.index_creator = IndexCreator(self.test_directory)

    def tearDown(self):
        for path in [
            'index.csv',
            'collection_term_count.pickle',
            'comment_term_count_dict.pickle',
            'compressed_index.csv',
            'compressed_seek_list.pickle',
            'seek_list.pickle',
            'symbol_encoding_pairs.pickle'
        ]:
            if os.path.isfile(f'{self.test_directory}/{path}'):
                os.remove(f'{self.test_directory}/{path}')


    def test_index_file(self):
        self.index_creator.create_index(compress_index=False)
        self.assertTrue(os.path.isfile(f'{self.test_directory}/index.csv'), 'index file was not created')
        with open(f'{self.test_directory}/index.csv', mode='r', encoding='utf-8') as index_file:
            self.assertMultiLineEqual(index_file.read(), expected_index, 'unexpected index content')

if __name__ == '__main__':
    unittest.main()
