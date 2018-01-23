#!/usr/bin/python3

import pickle
import heapq
import io
import os
import functools
import sys
import multiprocessing
import csv

import Stemmer
import nltk.tokenize

import Huffman
from Report import Report
from Common import CSVInputFile, Comment, posting_list_separator


def process_comments_file(directory, start_offset, end_offset,
                          comments_per_output_file=200000):
    assert(start_offset < end_offset)
    # process data between the given offsets
    comment_list = []
    file_number = 0
    with open(f'{directory}/sorted_comments.csv', mode='rb') as f:
        f.seek(start_offset)
        previous_offset = start_offset
        csv_reader = csv.reader(CSVInputFile(f), quoting=csv.QUOTE_ALL)

        tokenizer = nltk.tokenize.ToktokTokenizer()
        stemmer = Stemmer.Stemmer('english')
        stem = functools.lru_cache(None)(stemmer.stemWord)

        for csv_line in csv_reader:
            comment = (previous_offset, [])
            previous_offset = f.tell()
            comment_text_lower = csv_line[3].lower()
            for sentence in nltk.tokenize.sent_tokenize(comment_text_lower):
                for token in tokenizer.tokenize(sentence):
                    comment[1].append(stem(token))
            comment_list.append(comment)

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


def write_comments_to_temp_file(comment_list, file_name_prefix):
    print(f'writing {file_name_prefix}')
    all_comment_dict = {}
    term_count_dict = {}
    comment_term_count_dict = {}
    for comment in comment_list:

        comment_dict = {}
        comment_term_count_dict[comment[0]] = \
            len(comment[1])
        for position, stem in enumerate(comment[1]):
            if stem not in comment_dict.keys():
                comment_dict[stem] = [position]
            else:
                comment_dict[stem].append(position)
        for stem, positions in comment_dict.items():
            # positions = list of token pos in comment
            if stem not in all_comment_dict.keys():
                all_comment_dict[stem] = \
                    [(comment[0], positions)]
                term_count_dict[stem] = len(positions)
            else:
                all_comment_dict[stem].append(
                    (comment[0], positions))
                term_count_dict[stem] += len(positions)
    collection_term_count = \
        sum(comment_term_count_dict.values())

    with open(f'{file_name_prefix}_index.csv', mode='wb') as f:
        for stem in sorted(all_comment_dict.keys()):
            posting_list = all_comment_dict[stem]
            escaped_stem = stem.replace('"', '""')
            line_string = f'"{escaped_stem}"' \
                f'{posting_list_separator}{term_count_dict[stem]}'
            for posting_list_parts in sorted(posting_list):
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


def read_line_generator(file_path):
    with open(file_path) as target_file:
        line = target_file.readline().rstrip('\n')
        while line:
            yield line
            line = target_file.readline().rstrip('\n')


def create_list_from_csv(csv_file_path):
    result_list = []
    for line_number, line in enumerate(read_line_generator(csv_file_path)):
        parts = line.partition(',')
        assert(line_number == 0 or str(line_number) == parts[0])
        result_list.append(parts[2])
    return result_list


class IndexCreator():
    def __init__(self, directory):
        self.directory = directory
        assert(os.path.isfile(f'{self.directory}/sorted_comments.csv'))
        sys.setrecursionlimit(10000)
        self.report = Report(
            quiet_mode=__name__ != '__main__',
            log_file_path=f'{directory}/log_IndexCreator.py.csv')

    def create_index(self, skip_first_line=True, compress_index=True):
        # read csv to create comment_list

        with self.report.measure('processing sorted_comments.csv'):
            number_of_processes = os.cpu_count()
            self.report.report(f'starting {number_of_processes} processes')
            csv_size = os.stat(f'{self.directory}/sorted_comments.csv').st_size
            with multiprocessing.Pool(processes=number_of_processes) as pool:
                offsets = []
                with open(f'{self.directory}/sorted_comments.csv', mode='rb') \
                        as f:
                    if skip_first_line:
                        f.readline()
                        offsets.append(f.tell())
                    else:
                        offsets.append(0)

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

            self.partial_index_names = []
            for end_offset in offsets[1:]:
                file_path = \
                    f'{self.directory}/{end_offset}_file_number.pickle'
                with open(file_path, mode='rb') as f:
                    file_number = pickle.load(f)
                    for i in range(file_number):
                        self.partial_index_names.append(
                            f'{self.directory}/{end_offset}_{i}')
                os.remove(file_path)

            self.report.report(
                f'processed all comments')

        # merge indices
        with self.report.measure('merging index'):
            # comment term counts
            self.comment_term_count_dict = {}
            for file_prefix in self.partial_index_names:
                file_path = file_prefix + '_comment_term_count_dict.pickle'
                with open(file_path, mode='rb') as f:
                    self.comment_term_count_dict.update(pickle.load(f))
                os.remove(file_path)

            with open(f'{self.directory}/comment_term_count_dict.pickle',
                      mode='wb') as f:
                pickle.dump(self.comment_term_count_dict,
                            f, pickle.HIGHEST_PROTOCOL)

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
                        self.seek_list.append([current_min_term[1:-1].replace(
                            '""', '"'), current_offset])
                        current_offset += len(line_raw)

                    # reload lines where necessary
                    for key in current_active_indices:
                        linetest = index_files[key].readline().decode('utf-8')
                        if linetest == '':
                            # end of file
                            global_active_indices[key] = False
                            global_active_file_count -= 1
                            print('one file out,',
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

            # seek list
            with open(f'{self.directory}/seek_list.pickle', mode='wb') as f:
                pickle.dump(self.seek_list, f, pickle.HIGHEST_PROTOCOL)

            # cleanup
            for file in index_files:
                file.close()

            for file_prefix in self.partial_index_names:
                file_path = file_prefix + '_index.csv'
                os.remove(file_path)

        if compress_index:
            self.huffman_compression()

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
            with open(f'{self.directory}/compressed_index', mode='wb') \
                    as compressed_index_file:
                offset = 0
                for i, orig_line in enumerate(
                        read_line_generator(f'{self.directory}/index.csv'), 1):
                    term = next(csv.reader(io.StringIO(orig_line),
                                delimiter=posting_list_separator))[0]
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
    index_creator.report.all_time_measures()
