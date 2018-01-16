#!/usr/bin/python3

import pickle
import heapq
import io
import os
import functools
import sys
import multiprocessing

import Stemmer
import nltk.tokenize
# pip3 install git+https://github.com/provoke-vagueness/prefixtree
from prefixtree import PrefixDict

import Huffman
from Report import Report
from Common import *


def process_comments_file(directory, start_offset, end_offset):
    assert(start_offset < end_offset)
    # process data between the given offsets
    comment_list = []
    with open(f'{directory}/comments.csv', mode='rb') as f:
        f.seek(start_offset)
        previous_offset = start_offset
        csv_reader = csv.reader(CSVInputFile(f), quoting=csv.QUOTE_ALL)

        tokenizer = nltk.tokenize.ToktokTokenizer()
        stemmer = Stemmer.Stemmer('english')
        stem = functools.lru_cache(None)(stemmer.stemWord)

        for csv_line in csv_reader:
            comment = Comment().init_from_csv_line(csv_line, previous_offset)

            comment_text_lower = comment.text.lower()
            for sentence in nltk.tokenize.sent_tokenize(comment_text_lower):
                for token in tokenizer.tokenize(sentence):
                    comment.term_list.append(stem(token))
            comment_list.append(comment)

            previous_offset = f.tell()
            if start_offset == 0 and len(comment_list) % 5000 == 0:
                print(f'about {previous_offset / end_offset:7.2%} processed')

            if previous_offset == end_offset:
                break
            assert(previous_offset < end_offset)

    with open(f'{directory}/comment_list_{end_offset}.pickle',
              mode='wb') as f:
        pickle.dump(comment_list, f, pickle.HIGHEST_PROTOCOL)


class IndexCreator():
    def __init__(self, directory):
        self.directory = directory
        assert(os.path.isfile(f'{self.directory}/comments.csv'))
        sys.setrecursionlimit(10000)
        self.report = Report(
            quiet_mode=__name__ != '__main__',
            log_file_path=f'{directory}/log_IndexCreator.py.csv')

    def create_index(self, compress_index=True):
        # read csv to create comment_list
        with self.report.measure('processing comments.csv'):
            self.comment_list = []

            # number of usable CPUs
            number_of_processes = 4
            self.report.report(f'starting {number_of_processes} processes')
            csv_size = os.stat(f'{self.directory}/comments.csv').st_size
            with multiprocessing.Pool(processes=number_of_processes) as pool:
                offsets = [0]
                with open(f'{self.directory}/comments.csv', mode='rb') as f:
                    for i in range(1, number_of_processes + 1):
                        f.seek(int(i * csv_size / number_of_processes))
                        f.readline()
                        next_offset = f.tell()
                        if next_offset == offsets[-1]:
                            continue
                        offsets.append(next_offset)

                for start_offset, end_offset in zip(offsets, offsets[1:]):
                    pool.apply_async(
                        process_comments_file,
                        args=(self.directory, start_offset, end_offset))
                pool.close()
                pool.join()

            for end_offset in offsets[1:]:
                file_path = \
                    f'{self.directory}/comment_list_{end_offset}.pickle'
                with open(file_path, mode='rb') as f:
                    self.comment_list += pickle.load(f)
                os.remove(file_path)

            self.report.report(
                f'processed all {len(self.comment_list)} comments')

        # create index
        with self.report.measure('creating index'):
            all_comment_dict = {}
            term_count_dict = {}
            self.comment_term_count_dict = {}
            for comment in self.comment_list:
                comment_dict = {}
                self.comment_term_count_dict[comment.file_offset] = \
                    len(comment.term_list)
                for position, stem in enumerate(comment.term_list):
                    if stem not in comment_dict.keys():
                        comment_dict[stem] = [position]
                    else:
                        comment_dict[stem].append(position)
                for stem, positions in comment_dict.items():
                    # positions = list of token pos in comment
                    if stem not in all_comment_dict.keys():
                        all_comment_dict[stem] = \
                            [(comment.file_offset, positions)]
                        term_count_dict[stem] = len(positions)
                    else:
                        all_comment_dict[stem].append(
                            (comment.file_offset, positions))
                        term_count_dict[stem] += len(positions)
            self.collection_term_count = \
                sum(self.comment_term_count_dict.values())

        with self.report.measure('saving files'):
            # save index as csv
            self.seek_list = PrefixDict()
            current_offset = 0
            with open(f'{self.directory}/index.csv', mode='wb') as f:
                for stem in sorted(all_comment_dict.keys()):
                    posting_list = all_comment_dict[stem]
                    escaped_stem = stem.replace('"', '""')
                    line_string = f'"{escaped_stem}":{term_count_dict[stem]}'
                    for posting_list_part in sorted(posting_list):
                        # posting_list_part[0] = comment.file_offset
                        line_string += f':{posting_list_part[0]},'
                        # posting_list_part[1] =
                        # list of token positions in comment
                        line_string += ','.join(
                            (str(i) for i in posting_list_part[1]))
                    line_string += '\n'
                    line_raw = line_string.encode()
                    f.write(line_raw)
                    self.seek_list[stem] = current_offset
                    current_offset += len(line_raw)

            with open(f'{self.directory}/seek_list.pickle', mode='wb') as f:
                pickle.dump(self.seek_list, f, pickle.HIGHEST_PROTOCOL)

            with open(f'{self.directory}/comment_term_count_dict.pickle',
                      mode='wb') as f:
                pickle.dump(self.comment_term_count_dict,
                            f, pickle.HIGHEST_PROTOCOL)

            with open(f'{self.directory}/collection_term_count.pickle',
                      mode='wb') as f:
                pickle.dump(self.collection_term_count,
                            f, pickle.HIGHEST_PROTOCOL)

        if compress_index:
            self.huffman_compression()

        self.report.all_time_measures()

    def huffman_compression(self):
        # compress using Huffman encoding

        # count all occuring UTF-8 characters
        symbol_to_frequency_dict = {}
        with self.report.measure('counting utf8 characters'):
            with open(f'{self.directory}/index.csv') as index_file:
                chunk_size = 100000

                def next_chunk_generator():
                    chunk = index_file.read(chunk_size)
                    while chunk:
                        yield chunk
                        chunk = index_file.read(chunk_size)

                for i, chunk in enumerate(next_chunk_generator(), 1):
                    for symbol in chunk:
                        if symbol == '\n':
                            continue

                        if symbol not in symbol_to_frequency_dict.keys():
                            symbol_to_frequency_dict[symbol] = 1
                        else:
                            symbol_to_frequency_dict[symbol] += 1
                    self.report.progress(i, f' chunks counted ({chunk_size}'
                                         ' characters each)', 100)

        # derive huffman encoding from character counts
        with self.report.measure('deriving huffman encoding'):
            huffman_tree_root, symbol_to_encoding_dict = \
                Huffman.derive_encoding(symbol_to_frequency_dict)

        # save compressed index and corresponding seek_list
        with self.report.measure('saving compressed files'):
            with open(f'{self.directory}/huffman_tree.pickle', mode='wb') as f:
                pickle.dump(huffman_tree_root, f, pickle.HIGHEST_PROTOCOL)

            self.compressed_seek_list = {}
            with open(f'{self.directory}/index.csv') as index_file, \
                open(f'{self.directory}/compressed_index', mode='wb') \
                    as compressed_index_file:
                def read_line_generator():
                    orig_line = index_file.readline().rstrip('\n')
                    while orig_line:
                        yield orig_line
                        orig_line = index_file.readline().rstrip('\n')

                offset = 0
                for i, orig_line in enumerate(read_line_generator(), 1):
                    term = next(csv.reader(io.StringIO(orig_line),
                                delimiter=':'))[0]
                    line_without_term = orig_line[len(term) + 3:]
                    encoded_line = Huffman.encode(
                        line_without_term, symbol_to_encoding_dict)
                    compressed_index_file.write(encoded_line)

                    self.compressed_seek_list[term] = \
                        (offset, len(encoded_line))

                    self.report.progress(i, ' index lines compressed')

                    offset += len(encoded_line)

            with open(f'{self.directory}/compressed_seek_list.pickle',
                      mode='wb') as f:
                pickle.dump(
                    self.compressed_seek_list, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    data_directory = 'data/fake' if len(sys.argv) < 2 else sys.argv[1]
    index_creator = IndexCreator(data_directory)
    index_creator.create_index()
    # index_creator.huffman_compression()
    # with open(f'{data_directory}/huffman_tree.pickle',
    #           mode='rb') as huffman_tree_file, \
    #         open(f'{data_directory}/compressed_index',
    #              mode='rb') as compressed_index_file, \
    #         open(f'{data_directory}/compressed_seek_list.pickle',
    #              mode='rb') as compressed_seek_list_file:
    #     huffman_tree_root = pickle.load(huffman_tree_file)
    #     compressed_seek_list = pickle.load(compressed_seek_list_file)
    #     offset, length = compressed_seek_list['xi']
    #     compressed_index_file.seek(offset)
    #     binary_data = compressed_index_file.read(length)
    #     decoded_string = Huffman.decode(binary_data,
    #                                     huffman_tree_root)
    #     print(decoded_string)
