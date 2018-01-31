#!/usr/bin/env python3

import pickle
import io
import os
import functools
import sys
import multiprocessing
import csv
from collections import Counter

import numpy
import Stemmer
import nltk.tokenize
from dawg import RecordDAWG
from bitarray import bitarray as BitArray

import Huffman
from Report import Report
from Common import *


def process_comments_file(directory, start_offset, end_offset,
                          comments_per_output_file=200000):
    assert(start_offset < end_offset)
    # process data between the given offsets
    comment_list = []
    file_number = 0
    reply_to_index = {}
    cid_to_offset = {}
    with open(f'{directory}/comments.csv', mode='rb') as f:
        f.seek(start_offset)
        previous_offset = start_offset
        csv_reader = csv.reader(binary_read_line_generator(f))

        tokenizer = nltk.tokenize.ToktokTokenizer()
        stemmer = Stemmer.Stemmer('english')
        stem = functools.lru_cache(100)(stemmer.stemWord)

        for csv_line in csv_reader:
            if(not 6 <= len(csv_line) <= 8):
                print(f'WARNING: len(csv_line) == {len(csv_line)}',
                      'which is not between 6 and 8')
            cid = int(csv_line[0])

            cid_to_offset[cid] = previous_offset

            comment = (previous_offset, [])
            comment_text_lower = csv_line[3].lower()
            for sentence in nltk.tokenize.sent_tokenize(comment_text_lower):
                for token in tokenizer.tokenize(sentence):
                    comment[1].append(stemmer.stemWord(token))
            comment_list.append(comment)

            parent_cid = int(csv_line[5]) if csv_line[5] != '' else -1
            if parent_cid != -1:
                if parent_cid not in reply_to_index.keys():
                    reply_to_index[parent_cid] = [cid]
                else:
                    reply_to_index[parent_cid].append(cid)

            previous_offset = f.tell()
            if start_offset == 0 and len(comment_list) % 5000 == 0:
                print(f'about {previous_offset / end_offset:7.2%} processed')

            if len(comment_list) == comments_per_output_file \
                    or previous_offset == end_offset:
                write_comments_to_temp_file(
                    comment_list, f'{directory}/{end_offset}_{file_number}')
                file_number += 1
                comment_list = []
                if previous_offset == end_offset:
                    break

            assert(previous_offset < end_offset)

    with open(f'{directory}/{end_offset}_file_number.pickle', mode='wb') as f:
        pickle.dump(file_number, f, pickle.HIGHEST_PROTOCOL)

    with open(f'{directory}/{end_offset}_reply_to_index.pickle',
              mode='wb') as f:
        pickle.dump(reply_to_index, f, pickle.HIGHEST_PROTOCOL)

    with open(f'{directory}/{end_offset}_cid_to_offset.pickle',
              mode='wb') as f:
        pickle.dump(cid_to_offset, f, pickle.HIGHEST_PROTOCOL)


def write_comments_to_temp_file(comment_list, file_name_prefix):
    print(f'writing {file_name_prefix}')
    all_comment_dict = {}
    term_count_dict = {}
    comment_term_count_dict = {}
    for comment in comment_list:
        comment_dict = {}
        comment_term_count_dict[comment[0]] = len(comment[1])
        for position, stem in enumerate(comment[1]):
            if stem not in comment_dict.keys():
                comment_dict[stem] = [position]
            else:
                comment_dict[stem].append(position)
        for stem, positions in comment_dict.items():
            # positions = list of token pos in comment
            if stem not in all_comment_dict.keys():
                all_comment_dict[stem] = [(comment[0], positions)]
                term_count_dict[stem] = len(positions)
            else:
                all_comment_dict[stem].append((comment[0], positions))
                term_count_dict[stem] += len(positions)
    collection_term_count = sum(comment_term_count_dict.values())

    with open(f'{file_name_prefix}_index.csv', mode='wb') as f:
        for stem in sorted(all_comment_dict.keys()):
            if len(stem) > 1 and len(stem) <= 128:
                posting_list = all_comment_dict[stem]
                escaped_stem = stem.replace('"', '""')
                line_string = f'"{escaped_stem}"' \
                    f'{posting_list_separator}{term_count_dict[stem]}'
                for posting_list_parts in posting_list:
                    line_string += \
                        f'{posting_list_separator}{posting_list_parts[0]},'
                    # list of token positions in comment
                    line_string += ','.join(
                        (str(i) for i in posting_list_parts[1]))
                line_string += '\n'
                line_raw = line_string.encode()
                f.write(line_raw)

    with open(f'{file_name_prefix}_comment_term_count_dict.pickle',
              mode='wb') as f:
        pickle.dump(comment_term_count_dict, f, pickle.HIGHEST_PROTOCOL)

    with open(f'{file_name_prefix}_collection_term_count.pickle',
              mode='wb') as f:
        pickle.dump(collection_term_count, f, pickle.HIGHEST_PROTOCOL)


def create_list_from_csv(csv_file_path):
    result_list = []
    for line_number, line in enumerate(binary_read_line_generator_path(
            csv_file_path)):
        parts = line.partition(',')
        assert(line_number == 0 or str(line_number) == parts[0])
        result_list.append(parts[2])
    return result_list


class IndexCreator():
    def __init__(self, directory):
        self.directory = directory
        assert(os.path.isfile(f'{self.directory}/comments.csv'))
        sys.setrecursionlimit(10000)
        self.report = Report(
            quiet_mode=__name__ != '__main__',
            log_file_path=f'{directory}/log_IndexCreator.py.csv')

    def create_index(self):
        # read csv to create comment_list

        with self.report.measure('processing comments.csv'):
            number_of_processes = min(os.cpu_count(), 2)
            print(f'starting {number_of_processes} processes')
            csv_size = os.stat(f'{self.directory}/comments.csv').st_size
            with multiprocessing.Pool(processes=number_of_processes) as pool:
                offsets = [0]
                with open(f'{self.directory}/comments.csv', mode='rb') as f:
                    for i in range(1, number_of_processes + 1):
                        f.seek(int(i * csv_size / number_of_processes))
                        f.readline()
                        next_offset = f.tell()
                        if next_offset != offsets[-1]:
                            offsets.append(next_offset)

                def on_error(exception):
                    raise exception
                for start_offset, end_offset in zip(offsets, offsets[1:]):
                    pool.apply_async(
                        process_comments_file,
                        args=(self.directory, start_offset, end_offset),
                        error_callback=on_error)
                pool.close()
                pool.join()

            self.partial_index_names = []
            reply_to_index = {}
            cid_to_offset = {}
            for end_offset in offsets[1:]:
                file_number_path = \
                    f'{self.directory}/{end_offset}_file_number.pickle'
                with open(file_number_path, mode='rb') as f:
                    file_number = pickle.load(f)
                    for i in range(file_number):
                        self.partial_index_names.append(
                            f'{self.directory}/{end_offset}_{i}')
                os.remove(file_number_path)

                reply_to_index_part_path = f'{self.directory}/' \
                    f'{end_offset}_reply_to_index.pickle'
                with open(reply_to_index_part_path, mode='rb') as f:
                    reply_to_index_part = pickle.load(f)
                    for key, value in reply_to_index_part.items():
                        if key not in reply_to_index.keys():
                            reply_to_index[key] = value
                        else:
                            reply_to_index[key].extend(value)
                os.remove(reply_to_index_part_path)

                cid_to_offset_part_path = f'{self.directory}/' \
                    f'{end_offset}_cid_to_offset.pickle'
                with open(cid_to_offset_part_path, mode='rb') as f:
                    cid_to_offset_part = pickle.load(f)
                    cid_to_offset.update(cid_to_offset_part)
                os.remove(cid_to_offset_part_path)

            with open(f'{self.directory}/reply_to_index.pickle',
                      mode='wb') as f:
                pickle.dump(reply_to_index, f, pickle.HIGHEST_PROTOCOL)

            tempa = numpy.array([])
            ret = []
            ret2 = []
            for key in sorted(cid_to_offset.keys()):
                ret.append(numpy.int64(key))
                ret2.append(numpy.int64(cid_to_offset[key]))
            tempa = numpy.array(ret)
            numpy.save(f'{self.directory}/cids.npy', tempa)
            tempa2 = numpy.array(ret2)
            numpy.save(f'{self.directory}/comment_offsets_cid.npy', tempa2)

        # merge indices
        with self.report.measure('merging index'):
            # comment term counts
            self.comment_term_count_dict = {}
            for file_prefix in self.partial_index_names:
                file_path = file_prefix + '_comment_term_count_dict.pickle'
                with open(file_path, mode='rb') as f:
                    self.comment_term_count_dict.update(pickle.load(f))
                os.remove(file_path)
            tempa = numpy.array([])
            ret = []
            ret2 = []
            for key in sorted(self.comment_term_count_dict.keys()):
                ret.append(numpy.int64(key))
                ret2.append(numpy.int32(self.comment_term_count_dict[key]))
            tempa = numpy.array(ret)
            numpy.save(f'{self.directory}/comment_offsets.npy', tempa)
            tempa2 = numpy.array(ret2)
            numpy.save(f'{self.directory}/comment_term_counts.npy', tempa2)

            # collection term count
            self.collection_term_count = 0
            for file_prefix in self.partial_index_names:
                file_path = file_prefix + '_collection_term_count.pickle'
                with open(file_path, mode='rb') as f:
                    self.collection_term_count += pickle.load(f)
                os.remove(file_path)

            with open(f'{self.directory}/collection_term_count.pickle',
                      mode='wb') as f:
                pickle.dump(self.collection_term_count,
                            f, pickle.HIGHEST_PROTOCOL)

            # index
            index_files = []
            for file_prefix in self.partial_index_names:
                file_path = file_prefix + '_index.csv'
                index_files.append(open(file_path, mode='rb'))

            current_terms = []
            current_meta = []
            current_posting_lists = []
            global_active_indices = []
            global_active_file_count = 0

            for file in index_files:
                line = file.readline().decode('utf-8').rstrip('\n').split(
                    posting_list_separator, 2)
                current_terms.append(line[0])
                current_meta.append(int(line[1]))
                current_posting_lists.append(line[2])
                global_active_indices.append(True)
                global_active_file_count += 1

            current_active_indices = []
            current_min_term = None
            self.seek_list = []
            current_offset = 0
            terms_done = 0

            with open(f'{self.directory}/index.csv', mode='wb') as f:
                while global_active_file_count > 0:
                    # find next term to write
                    for key, term in enumerate(current_terms):
                        if not global_active_indices[key]:
                            continue
                        if current_min_term is None or term < current_min_term:
                            current_active_indices = [key]
                            current_min_term = term
                        elif term == current_min_term:
                            current_active_indices.append(key)

                    # merge all lines containing term

                    if len(current_min_term) <= 128:
                        meta = 0
                        for key in current_active_indices:
                            meta += current_meta[key]

                        line_string = \
                            f'{current_min_term}{posting_list_separator}{meta}'
                        for key in current_active_indices:
                            line_string += f'{posting_list_separator}' \
                                f'{current_posting_lists[key]}'

                        line_string += '\n'
                        line_raw = line_string.encode()
                        f.write(line_raw)
                        term = current_min_term[1:-1].replace('""', '"')
                        self.seek_list.append((term, [current_offset]))
                        current_offset += len(line_raw)

                    # reload lines where necessary
                    for key in current_active_indices:
                        linetest = index_files[key].readline().decode('utf-8')
                        if linetest == '':
                            # end of file
                            global_active_indices[key] = False
                            global_active_file_count -= 1
                            print('one file out, '
                                  f'{global_active_file_count} remaining')
                        else:
                            line = linetest.rstrip('\n').split(
                                posting_list_separator, 2)
                            current_terms[key] = line[0]
                            current_meta[key] = int(line[1])
                            current_posting_lists[key] = line[2]

                    current_min_term = None
                    current_active_indices = []
                    terms_done += 1
                    if terms_done % 100000 == 0:
                        print(f'Merged {terms_done} terms.')

            self.seek_list = RecordDAWG('>Q', self.seek_list)
            self.seek_list.save(f'{self.directory}/seek_list.dawg')

            for f in index_files:
                f.close()

            for file_prefix in self.partial_index_names:
                file_path = file_prefix + '_index.csv'
                os.remove(file_path)

        self.huffman_compression(generate_encoding=False)

        with self.report.measure('processing authors & articles'):
            with open(f'{self.directory}/authors_list.pickle', mode='wb') as f:
                pickle.dump(
                    create_list_from_csv(f'{self.directory}/authors.csv'),
                    f, pickle.HIGHEST_PROTOCOL)

            with open(f'{self.directory}/articles_list.pickle', mode='wb') \
                    as f:
                pickle.dump(
                    create_list_from_csv(f'{self.directory}/articles.csv'),
                    f, pickle.HIGHEST_PROTOCOL)

    def huffman_compression(self, generate_encoding=False):
        # compress using Huffman encoding
        symbol_to_encoding_dict = {}

        # count all occuring UTF-8 characters
        if generate_encoding:
            symbol_to_frequency_dict = Counter()
            with self.report.measure('counting utf8 characters'):
                with open(f'{self.directory}/index.csv') as index_file:
                    chunk_size = 100000

                    def next_chunk_generator():
                        chunk = index_file.read(chunk_size)
                        while chunk:
                            yield chunk
                            chunk = index_file.read(chunk_size)

                    for i, chunk in enumerate(next_chunk_generator(), 1):
                        symbol_to_frequency_dict.update(Counter(chunk))
                        self.report.progress(
                            i, f' chunks counted ({chunk_size} characters '
                            'each)', 100)
                if '\n' in symbol_to_frequency_dict.keys():
                    del symbol_to_frequency_dict['\n']

            # derive huffman encoding from character counts
            with self.report.measure('deriving huffman encoding'):
                symbol_to_encoding_dict = Huffman.derive_encoding(
                    symbol_to_frequency_dict)
            for key, value in symbol_to_encoding_dict.items():
                assert(len(key) == 1)
                symbol_to_encoding_list[ord(key[0])] = value
            with open(f'{self.directory}/symbol_to_encoding_dict.pickle',
                      mode='wb') as f:
                pickle.dump(symbol_to_encoding_dict, f,
                            pickle.HIGHEST_PROTOCOL)
        else:
            # optimal encoding for guardian
            # character distribution should be similar for all datasets
            symbol_to_encoding_dict = {
                '\a': BitArray('1111'), ',': BitArray('001'),
                '0': BitArray('1000'), '1': BitArray('011'),
                '2': BitArray('010'), '3': BitArray('000'),
                '4': BitArray('1110'), '5': BitArray('1101'),
                '6': BitArray('1100'), '7': BitArray('1011'),
                '8': BitArray('1010'), '9': BitArray('1001')
            }

        with open(f'{self.directory}/symbol_to_encoding_dict.pickle',
                  mode='wb') as f:
            pickle.dump(symbol_to_encoding_dict, f, pickle.HIGHEST_PROTOCOL)

        # save compressed index and corresponding seek_list
        with self.report.measure('saving compressed files'):
            self.compressed_seek_list = []
            with open(f'{self.directory}/compressed_index', mode='wb') \
                    as compressed_index_file:
                offset = 0
                for i, orig_line in enumerate(binary_read_line_generator_path(
                        f'{self.directory}/index.csv'), 1):
                    term = next(csv.reader(io.StringIO(orig_line),
                                delimiter=posting_list_separator))[0]
                    line_without_term = orig_line[len(term) + 3:]
                    encoded_line = Huffman.encode(
                        line_without_term, symbol_to_encoding_dict)
                    compressed_index_file.write(encoded_line)

                    self.compressed_seek_list.append(
                        (term, (offset, len(encoded_line))))

                    self.report.progress(i, ' index lines compressed', 100000)

                    offset += len(encoded_line)
            with open(f'{self.directory}/compressed_seek_list.pickle',
                      mode='wb') as f:
                pickle.dump(self.compressed_seek_list, f,
                            pickle.HIGHEST_PROTOCOL)
            self.compressed_seek_list = \
                RecordDAWG('>QQ', self.compressed_seek_list)
            self.compressed_seek_list.save(
                f'{self.directory}/compressed_seek_list.dawg')


if __name__ == '__main__':
    data_directory = 'data/fake' if len(sys.argv) < 2 else sys.argv[1]
    index_creator = IndexCreator(data_directory)
    index_creator.create_index()
    index_creator.report.all_time_measures()
