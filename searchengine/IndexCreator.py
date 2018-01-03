#!/usr/bin/python3

import Stemmer
from nltk.tokenize import ToktokTokenizer, sent_tokenize
from collections import OrderedDict
import pickle
import heapq
import io
import os
from sys import argv
from prefixtree import PrefixDict # pip3 install git+https://github.com/provoke-vagueness/prefixtree

import Report
from Common import *
from Huffman import *

if __name__ != '__main__':
    Report.set_quiet_mode(True)

class IndexCreator():
    def __init__(self, directory):
        self.directory = directory
        assert(os.path.isfile(f'{self.directory}/comments.csv'))

    def create_index(self, compress_index=True):
        #read csv to create comment_list
        Report.begin('parsing comments.csv')
        self.comment_list = []
        with open(f'{self.directory}/comments.csv', mode='rb') as f:
            csv_reader = csv.reader(CSVInputFile(f), quoting=csv.QUOTE_ALL)
            previous_offset = f.tell()
            for csv_line in csv_reader:
                comment = Comment().init_from_csv_line(csv_line, previous_offset)
                self.comment_list.append(comment)
                Report.progress(len(self.comment_list), ' comments parsed')
                previous_offset = f.tell()
        Report.report(f'found {len(self.comment_list)} comments')
        Report.finish('parsing comments.csv')

        #process comments (tokenize and stem tokens)
        Report.begin('processing comments')
        stemmer = Stemmer.Stemmer('english')
        tokenizer = ToktokTokenizer()
        comments_processed = 0
        for comment in self.comment_list:
            comment_text_lower = comment.text.lower()
            comment.term_list = []
            for sentence in sent_tokenize(comment_text_lower):
                tokens = tokenizer.tokenize(sentence)
                comment.term_list += stemmer.stemWords(tokens)
            comments_processed += 1
            Report.progress(comments_processed, f'/{len(self.comment_list)} comments processed')
        Report.report(f'{comments_processed}/{len(self.comment_list)} comments processed')
        Report.finish('processing comments')

        #create index
        Report.begin('creating index')
        all_comment_dict = {}
        term_count_dict = {}
        self.comment_term_count_dict = {}
        self.collection_term_count = 0
        for comment in self.comment_list:
            position = 0
            comment_dict = {}
            self.comment_term_count_dict[comment.file_offset] = len(comment.term_list)
            self.collection_term_count += len(comment.term_list)
            for stem in comment.term_list:
                if not stem in comment_dict:
                    comment_dict[stem] = []
                comment_dict[stem].append(position)
                position += 1
            for stem, positions in comment_dict.items():
                # positions = list of token pos in comment
                if not stem in all_comment_dict:
                    all_comment_dict[stem] = []
                    term_count_dict[stem] = 0
                all_comment_dict[stem].append([comment.file_offset, positions])
                term_count_dict[stem] += len(positions)
        Report.finish('creating index')

        #save index as csv
        Report.begin('saving files')
        sorted_all_comment_dict = OrderedDict(sorted(all_comment_dict.items(), key=lambda t:t[0]))
        self.seek_list = PrefixDict()
        current_offset = 0
        with open(f'{self.directory}/index.csv', mode='wb') as f:
            for stem, posting_list in sorted_all_comment_dict.items():
                escaped_stem = stem.replace('"', '""')
                line_string = f'"{escaped_stem}":{term_count_dict[stem]}'
                sorted_posting_list = [x for x in sorted(posting_list)]
                for posting_list_part in sorted_posting_list:
                    line_string += ':'
                    line_string += str(posting_list_part[0])
                    for position in posting_list_part[1]:
                        line_string += ','
                        line_string += str(position)
                line_string += '\n'
                line_raw = line_string.encode()
                f.write(line_raw)
                self.seek_list[stem] = current_offset
                current_offset += len(line_raw)

        #pickle out seek_list
        with open(f'{self.directory}/seek_list.pickle', mode='wb') as f:
            pickle.dump(self.seek_list, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{self.directory}/comment_term_count_dict.pickle', mode='wb') as f:
            pickle.dump(self.comment_term_count_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{self.directory}/collection_term_count.pickle', mode='wb') as f:
            pickle.dump(self.collection_term_count, f, pickle.HIGHEST_PROTOCOL)

        Report.finish('saving files')

        if compress_index:
            self.huffman_compression()

        Report.all_time_measures()

    def huffman_compression(self):
        #compress using Huffman encoding

        #count all occuring UTF-8 characters
        Report.begin('counting utf8 characters')
        symbol_to_frequency_dict = {}
        with open(f'{self.directory}/index.csv', mode='r', encoding='utf-8') as index_file:
            def get_next_chunk(chunk_size = 100000):
                """Reads 100000 character from the given textfile"""
                chunk = index_file.read(chunk_size)
                while chunk:
                    yield chunk
                    chunk = index_file.read(chunk_size)
            i = 0
            for chunk in get_next_chunk():
                for symbol in chunk:
                    if symbol == '\n':
                        continue

                    if not symbol in symbol_to_frequency_dict.keys():
                        symbol_to_frequency_dict[symbol] = 1
                    else:
                        symbol_to_frequency_dict[symbol] += 1
                    i += 1
                    Report.progress(i, ' characters counted', 2500000)
        Report.finish('counting utf8 characters')

        # derive huffman encoding from character counts
        Report.begin('deriving huffman encoding')
        huffman_tree_root, symbol_to_encoding_dict = derive_huffman_encoding(symbol_to_frequency_dict)
        Report.finish('deriving huffman encoding')

        Report.begin('saving compressed files')
        with open(f'{self.directory}/huffman_tree.pickle', mode='wb') as f:
            pickle.dump(huffman_tree_root, f, pickle.HIGHEST_PROTOCOL)

        self.compressed_seek_list = {}
        with open(f'{self.directory}/index.csv', mode='r', encoding='utf-8') as index_file:
            with open(f'{self.directory}/compressed_index', mode='wb') as compressed_index_file:
                orig_line = index_file.readline().rstrip('\n')
                offset = 0
                i = 0
                while orig_line:
                    term = next(csv.reader(io.StringIO(orig_line), delimiter=':'))[0]
                    line_without_term = orig_line[len(term) + 3:]
                    encoded_line = encode_huffman(line_without_term, symbol_to_encoding_dict)
                    compressed_index_file.write(encoded_line)

                    self.compressed_seek_list[term] = (offset, len(encoded_line))

                    i += 1
                    Report.progress(i, ' index lines compressed')

                    offset += len(encoded_line)
                    orig_line = index_file.readline().rstrip('\n')

        with open(f'{self.directory}/compressed_seek_list.pickle', mode='wb') as f:
            pickle.dump(self.compressed_seek_list, f, pickle.HIGHEST_PROTOCOL)
        Report.finish('saving compressed files')

if __name__ == '__main__':
    data_directory = 'data/fake' if len(argv) < 2 else argv[1]
    index_creator = IndexCreator(data_directory)
    index_creator.create_index()
    # index_creator.huffman_compression()
    # with open(f'{data_directory}/huffman_tree.pickle', mode='rb') as huffman_tree_file:
    #     with open(f'{data_directory}/compressed_index', mode='rb') as compressed_index_file:
    #         with open(f'{data_directory}/compressed_seek_list.pickle', mode='rb') as compressed_seek_list_file:
    #             huffman_tree_root = pickle.load(huffman_tree_file)
    #             compressed_seek_list = pickle.load(compressed_seek_list_file)
    #             offset, length = compressed_seek_list['xi']
    #             compressed_index_file.seek(offset)
    #             binary_data = compressed_index_file.read(length)
    #             decoded_string = decode_huffman(binary_data, huffman_tree_root)
    #             print(decoded_string)
