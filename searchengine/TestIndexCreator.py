#!/usr/bin/python3

import unittest
import filecmp

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
            'compressed_index',
            'compressed_seek_list.pickle',
            'seek_list.pickle',
            'symbol_encoding_pairs.pickle',
            'huffman_tree.pickle'
        ]:
            if os.path.isfile(f'{self.test_directory}/{path}'):
                os.remove(f'{self.test_directory}/{path}')


    def test_index_file(self):
        self.assertTrue(os.path.isfile(f'{self.test_directory}/expected_index.csv'),
            f'could not find expected_index.csv in {self.test_directory}')
        expected_comment_list = [
            Comment(1767167970, 'http://en.people.cn/n/2015/0101/c90785-8830442.html', 'klive',
                '2015-01-01T14:34:08', None, 0, 0, 'Tragic event', 0, ['tragic', 'event']),
            Comment(1766936418, 'http://en.people.cn/n/2015/0101/c90785-8830442.html', 'Wang Wei',
                '2015-01-01T06:24:06', None, 0, 0, 'Xi > TrumP', 127, ['xi', '>', 'trump']),
            Comment(1766866409, 'http://en.people.cn/n/2015/0101/c90785-8830442.html', 'enpeople',
                '2015-01-01T04:18:20', None, 0, 0, 'some special §¸…· characters', 255,
                ['some', 'special', '§¸', '…', '·', 'charact'])
        ]
        self.assertEqual(self.index_creator.comment_list, expected_comment_list)
        self.assertTrue(os.path.isfile(f'{self.test_directory}/index.csv'),
            'index.csv was not created')
        self.assertTrue(filecmp.cmp(f'{self.test_directory}/index.csv',
            f'{self.test_directory}/expected_index.csv'),
            'expected and created index.csv are different')

    def test_seek_list(self):
        expected_seek_list = [ ('>', 0), ('charact', 12), ('event', 30), ('some', 44),
            ('special', 59), ('tragic', 77), ('trump', 92), ('xi', 108), ('§¸', 121),
            ('·', 136), ('…', 149) ]
        self.assertEqual(self.index_creator.seek_list, expected_seek_list)

    def test_term_counts(self):
        expected_comment_term_count_dict = {0: 2, 127: 3, 255: 6}
        self.assertDictEqual(self.index_creator.comment_term_count_dict,
            expected_comment_term_count_dict)
        self.assertEqual(self.index_creator.collection_term_count, 11)

    def test_compressed_seek_list(self):
        self.index_creator.huffman_compression()
        expected_compressed_seek_list = { '>': (0, 6), 'charact': (6, 9), 'event': (15, 8),
            'some': (23, 8), 'special': (31, 10), 'tragic': (41, 8), 'trump': (49, 9),
            'xi': (58, 7), '§¸': (65, 6), '·': (71, 6), '…': (77, 6)}
        self.assertDictEqual(self.index_creator.compressed_seek_list, expected_compressed_seek_list)

    def test_compressed_index_file(self):
        self.assertTrue(os.path.isfile(f'{self.test_directory}/expected_compressed_index'),
            f'could not find expected_compressed_index file in {self.test_directory}')
        self.index_creator.huffman_compression()
        self.assertTrue(os.path.isfile(f'{self.test_directory}/compressed_index'),
            'compressed index file was not created')
        self.assertTrue(filecmp.cmp(f'{self.test_directory}/compressed_index',
            f'{self.test_directory}/expected_compressed_index'),
            'expected and created compressed_index files are different')

if __name__ == '__main__':
    unittest.main()
