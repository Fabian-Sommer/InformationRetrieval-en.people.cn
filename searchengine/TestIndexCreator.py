#!/usr/bin/python3

import unittest

from IndexCreator import *

class TestIndexCreation(unittest.TestCase):
    def setUp(self):
        # self.maxDiff = None
        self.test_directory = 'data/test'
        self.index_creator = IndexCreator(self.test_directory)
        self.index_creator.create_index(compress_index=False)

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
        expected_comment_list = [
            Comment(1767167970, 'http://en.people.cn/n/2015/0101/c90785-8830442.html', 'klive', '2015-01-01T14:34:08', None, 0, 0, 'Tragic event', 0, ['tragic', 'event']),
            Comment(1766936418, 'http://en.people.cn/n/2015/0101/c90785-8830442.html', 'Wang Wei', '2015-01-01T06:24:06', None, 0, 0, 'Xi > TrumP', 127, ['xi', '>', 'trump']),
            Comment(1766866409, 'http://en.people.cn/n/2015/0101/c90785-8830442.html', 'enpeople', '2015-01-01T04:18:20', None, 0, 0, 'some special §¸…· characters', 255, ['some', 'special', '§¸…·', 'charact'])
        ]
        self.assertEqual(self.index_creator.comment_list, expected_comment_list, "unexpected comment list")
        self.assertTrue(os.path.isfile(f'{self.test_directory}/index.csv'), 'index file was not created')
        expected_index = r'''">":1:127,1
"charact":1:255,3
"event":1:0,1
"some":1:255,0
"special":1:255,1
"tragic":1:0,0
"trump":1:127,2
"xi":1:127,0
"§¸…·":1:255,2
'''
        with open(f'{self.test_directory}/index.csv', mode='r', encoding='utf-8') as index_file:
            self.assertMultiLineEqual(index_file.read(), expected_index, 'unexpected index content')

    def test_seek_list(self):
        expected_seek_list = [('>', 0), ('charact', 12), ('event', 30), ('some', 44),
            ('special', 59), ('tragic', 77), ('trump', 92), ('xi', 108), ('§¸…·', 121)]
        self.assertEqual(self.index_creator.seek_list, expected_seek_list, 'unexpected seek list')



if __name__ == '__main__':
    unittest.main()
