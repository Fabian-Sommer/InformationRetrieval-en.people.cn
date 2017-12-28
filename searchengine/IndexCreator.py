#!/usr/bin/python3

import Stemmer
import nltk
from collections import OrderedDict
import pickle
import heapq
import io
import os

from Common import *

def report_progress(progress, message):
    if progress % 1000 == 0:
        print(f'{progress}{message}')

# used for index compression
def huffman_encode(symb2freq):
    """Huffman encode the given dict mapping symbols to weights"""
    heap = [[wt, [sym, '']] for sym, wt in symb2freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

class IndexCreator():
    def __init__(self, directory):
        self.directory = directory
        assert(os.path.isfile(f'{self.directory}/comments.csv'))

    def create_comment_list(self):
        #read csv
        self.comment_list = []
        with open(f'{self.directory}/comments.csv', mode='rb') as f:
            csv_reader = csv.reader(CSVInputFile(f), quoting=csv.QUOTE_ALL)
            previous_offset = f.tell()
            for row in csv_reader:
                comment = Comment()
                comment.cid = int(row[0])
                comment.url = row[1]
                comment.author = row[2]
                comment.time = row[3]
                comment.parent = None if row[4] == 'None' else int(row[4])
                comment.likes = int(row[5])
                comment.dislikes = int(row[6])
                comment.text = row[7]
                comment.file_offset = previous_offset
                self.comment_list.append(comment)
                report_progress(len(self.comment_list), ' comments parsed')
                previous_offset = f.tell()
        print(f'parsed csv into {len(self.comment_list)} comments.')

        #process comments (tokenize and stem tokens)
        stemmer = Stemmer.Stemmer('english')
        comments_processed = 0
        for comment in self.comment_list:
            raw_tokens = nltk.word_tokenize(comment.text.lower())
            comment.term_list = stemmer.stemWords(raw_tokens)
            comments_processed += 1
            report_progress(comments_processed, f'/{len(self.comment_list)} comments processed')
        print(f'{comments_processed}/{len(self.comment_list)} comments processed - done')

    def create_index(self, compress_index=True):
        self.create_comment_list()

        #create index
        all_comment_dict = {}
        self.term_count_dict = {}
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
                if not stem in all_comment_dict:
                    all_comment_dict[stem] = []
                    self.term_count_dict[stem] = 0
                all_comment_dict[stem].append([comment.file_offset, positions])
                self.term_count_dict[stem] += len(positions)
        # positions = list of token pos in comment

        #save index as csv
        self.sorted_all_comment_dict = OrderedDict(sorted(all_comment_dict.items(), key=lambda t:t[0]))
        offset_dict = {}
        current_offset = 0
        with open(f'{self.directory}/index.csv', mode='wb') as f:
            for stem, posting_list in self.sorted_all_comment_dict.items():
                escaped_stem = stem.replace('"', '""')
                line_string = f'"{escaped_stem}":{self.term_count_dict[stem]}'
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
                offset_dict[stem] = current_offset
                current_offset += len(line_raw)
        self.seek_list = [(k, offset_dict[k]) for k in sorted(offset_dict)]


        #pickle out seek_list (sorted offset_dict)
        with open(f'{self.directory}/seek_list.pickle', mode='wb') as f:
            pickle.dump(self.seek_list, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{self.directory}/comment_term_count_dict.pickle', mode='wb') as f:
            pickle.dump(self.comment_term_count_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{self.directory}/collection_term_count.pickle', mode='wb') as f:
            pickle.dump(self.collection_term_count, f, pickle.HIGHEST_PROTOCOL)

        if compress_index:
            print('starting compression...')
            self.huffman_compression()
        print('index creation done')

    def huffman_compression(self):
        #compress using Huffman encoding
        #count all occuring UTF-8 characters
        character_counts = {}
        with open(f'{self.directory}/index.csv', mode='r', encoding='utf-8') as f:
            i = 0
            def get_next_character(f):
                """Reads one character from the given textfile"""
                c = f.read(1)
                while c:
                    yield c
                    c = f.read(1)

            for c in get_next_character(f):
                i += 1
                if i%1000000 == 0:
                    print(f'{int(i/1000000)} MB counted')
                if c != '\n':
                    if not c in character_counts:
                        character_counts[c] = 1
                    else:
                        character_counts[c] += 1
        symbol_encoding_pairs = huffman_encode(character_counts)
        with open(f'{self.directory}/symbol_encoding_pairs.pickle', mode='wb') as f:
            pickle.dump(symbol_encoding_pairs, f, pickle.HIGHEST_PROTOCOL)
        symbol_encoding = {}
        # TODO: for key, value ...
        for pair in symbol_encoding_pairs:
            symbol_encoding[pair[0]] = pair[1]

        compressed_seek_list = {}
        offset = 0
        with open(f'{self.directory}/index.csv', mode='r', encoding='utf-8') as inf:
            with open(f'{self.directory}/compressed_index.csv', mode='wb') as of:
                orig_line = inf.readline().rstrip('\n')
                i = 0
                while orig_line:
                    i += 1
                    new_line = ''
                    for c in orig_line:
                        new_line += symbol_encoding[c]
                    padding = (8 - (len(new_line) % 8)) % 8
                    assert(0 <= padding and padding < 8)
                    new_line += padding * '0'
                    assert(len(new_line) % 8 == 0)
                    # first split into 8-bit chunks
                    bit_strings = [new_line[i:i + 8] for i in range(0, len(new_line), 8)]
                    # then convert to integers
                    byte_list = [int(b, 2) for b in bit_strings]
                    of.write(str(padding).encode())
                    of.write(bytearray(byte_list))
                    cs = csv.reader(io.StringIO(orig_line), delimiter=':')
                    term = ''
                    for csv_result in cs:
                        term = csv_result[0]
                        break
                    compressed_seek_list[term] = [offset, 1 + len(byte_list)]
                    offset += 1 + len(byte_list)
                    orig_line = inf.readline().rstrip('\n')

        with open(f'{self.directory}/compressed_seek_list.pickle', mode='wb') as f:
            pickle.dump(compressed_seek_list, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    IndexCreator('data/fake').create_index()
