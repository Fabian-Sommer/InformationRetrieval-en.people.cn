#!/usr/bin/python3

import csv
import nltk
import Stemmer
from nltk.corpus import stopwords
import string
import pickle
import math
from collections import OrderedDict
import random
import re
import heapq
import time
import array
import io

class Comment():
    cid = 0
    url = ''
    author = ''
    time = ''
    parent = 0
    likes = 0
    dislikes = 0
    text = ''
    file_offset = 0
    term_list = []

class CSVInputFile(object):
    """ File-like object. """
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.offset = None
        self.linelen = None

    def __iter__(self):
        return self

    def __next__(self):
        offset = self.csv_file.tell()
        data = self.csv_file.readline()
        if not data:
            raise StopIteration
        self.offset = offset
        self.linelen = len(data)
        return data.decode()

    next = __next__

class SearchEngine():

    def __init__(self):
        self.seek_list = []
        self.comment_file = None
        self.index_file = None
        self.comment_csv_reader = None
        self.comment_term_count_dict = None
        self.collection_term_count = 0
        self.stemmer = Stemmer.Stemmer('english')

    def index(self, directory):
        #read csv
        comment_list = []
        with open(f'{directory}/comments.csv', mode='rb') as f:
            csv_reader = csv.reader(CSVInputFile(f), quoting=csv.QUOTE_ALL)
            last_offset = 0
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
                comment.file_offset = last_offset
                last_offset = f.tell()
                comment_list.append(comment)

        print(f'Parsed csv into {len(comment_list)} comments.')

        #process comments (tokenize, remove stopwords, stem tokens)
        comments_processed = 0
        # phrase queries need stopwords...
        # stops = set(stopwords.words('english') + list(string.punctuation))
        for comment in comment_list:
            raw_tokens = nltk.word_tokenize(comment.text.lower())
            comment.term_list = self.stemmer.stemWords(raw_tokens)
            comments_processed += 1
            if comments_processed % 1000 == 0:
                print(f'{comments_processed}/{len(comment_list)} comments processed')
        print(f'{comments_processed}/{len(comment_list)} comments processed - done')

        #create index
        all_comment_dict = {}
        term_count_dict = {}
        comment_term_count_dict = {}
        collection_term_count = 0
        for comment in comment_list:
            position = 0
            comment_dict = {}
            comment_term_count_dict[comment.file_offset] = len(comment.term_list)
            collection_term_count += len(comment.term_list)
            for stem in comment.term_list:
                if not stem in comment_dict:
                    comment_dict[stem] = []
                comment_dict[stem].append(position)
                position += 1
            for stem, positions in comment_dict.items():
                if not stem in all_comment_dict:
                    all_comment_dict[stem] = []
                    term_count_dict[stem] = 0
                all_comment_dict[stem].append([comment.file_offset, positions])
                term_count_dict[stem] += len(positions)
        # positions = list of token pos in comment, ignoring stopwords -> todo include stopwords
        sorted_all_comment_dict = OrderedDict(sorted(all_comment_dict.items(), key=lambda t:t[0]))

        #save index as csv
        offset_dict = {}
        current_offset = 0
        with open(f'{directory}/index.csv', mode='wb') as f:
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
                offset_dict[stem] = current_offset
                current_offset += len(line_raw)

        #seek list should be a sorted list
        seek_list = [(k, offset_dict[k]) for k in sorted(offset_dict)]

        #pickle out offset_dict
        with open(f'{directory}/seek_list.pickle', mode='wb') as f:
            pickle.dump(seek_list, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{directory}/comment_term_count_dict.pickle', mode='wb') as f:
            pickle.dump(comment_term_count_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(f'{directory}/collection_term_count.pickle', mode='wb') as f:
            pickle.dump(collection_term_count, f, pickle.HIGHEST_PROTOCOL)

    def get_next_character(self, f):
        """Reads one character from the given textfile"""
        c = f.read(1)
        while c:
            yield c
            c = f.read(1)

    def encode(self, symb2freq):
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

    def compressIndex(self, directory):
        #compress using Huffman encoding
        #count all occuring UTF-8 characters
        character_count = {}
        i = 0
        with open(f'{directory}/index.csv', mode='r', encoding='utf-8') as f:
            for c in self.get_next_character(f):
                i += 1
                if i%1000000 == 0:
                    print(f'{int(i/1000000)} MB counted')
                if c != '\n':
                    if not c in character_count:
                        character_count[c] = 1
                    else:
                        character_count[c] += 1
        symbol_encoding_pairs = self.encode(character_count)
        with open(f'{directory}/symbol_encoding_pairs.pickle', mode='wb') as f:
            pickle.dump(symbol_encoding_pairs, f, pickle.HIGHEST_PROTOCOL)
        symbol_encoding = {}
        for pair in symbol_encoding_pairs:
            symbol_encoding[pair[0]] = pair[1]

        compressed_seek_list = {}
        offset = 0
        with open(f'{directory}/index.csv', mode='r', encoding='utf-8') as inf:
            with open(f'{directory}/compressed_index.csv', mode='wb') as of:
                orig_line = inf.readline().rstrip('\n')
                i = 0
                while orig_line:
                    i += 1
                    print(i)
                    new_line = ''
                    for c in orig_line:
                        new_line += symbol_encoding[c]
                    padding = (8 - (len(new_line) % 8)) % 8
                    assert(padding < 8)
                    assert(padding >= 0)
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
        print(str(offset))
        with open(f'{directory}/compressed_seek_list.pickle', mode='wb') as f:
            pickle.dump(compressed_seek_list, f, pickle.HIGHEST_PROTOCOL)





    def loadCompressedIndex(self, directory):
        #TODO
        print('unimplemented loadCompressedIndex was called')

    def loadIndex(self, directory):
        with open(f'{directory}/seek_list.pickle', mode='rb') as f:
            self.seek_list = pickle.load(f)
        with open(f'{directory}/comment_term_count_dict.pickle', mode='rb') as f:
            self.comment_term_count_dict = pickle.load(f)
        with open(f'{directory}/collection_term_count.pickle', mode='rb') as f:
            self.collection_term_count = pickle.load(f)
        self.comment_file = open(f'{directory}/comments.csv', mode='rb')
        self.index_file = open(f'{directory}/index.csv', mode='r', encoding='utf-8')
        self.comment_csv_reader = csv.reader(CSVInputFile(self.comment_file), quoting=csv.QUOTE_ALL)

    # returns score for ranking based on natural language model with dirichlet smoothing
    # query_terms: list of query terms, stemmed and filtered
    # comment_offsets: list of offsets of comments into comment file
    def get_dirichlet_smoothed_score(self, query_terms, comment_offsets, mu = 1500):
        score_list = [0 for x in comment_offsets]
        for query_term in query_terms:
            i = self.get_index_in_seek_list(query_term)
            if i == -1:
                next
            self.index_file.seek(self.seek_list[i][1])
            posting_list = self.index_file.readline().rstrip('\n')
            posting_list_parts = posting_list.split(':')
            c_query_term = int(posting_list_parts[1])
            comment_offsets_index = 0
            for comment_list in posting_list_parts[2:]:
                if comment_offsets_index >= len(comment_offsets):
                    break
                occurences = comment_list.split(',')
                while comment_offsets_index < len(comment_offsets) and int(occurences[0]) > comment_offsets[comment_offsets_index]:
                    #term not found -> 0 occurences in comment
                    score_list[comment_offsets_index] += math.log(((mu * c_query_term / self.collection_term_count))/(self.comment_term_count_dict[comment_offsets[comment_offsets_index]] + mu))
                    comment_offsets_index += 1

                if comment_offsets_index < len(comment_offsets) and int(occurences[0]) == comment_offsets[comment_offsets_index]:
                    fD_query_term = len(occurences) - 1
                    score_list[comment_offsets_index] += math.log((fD_query_term + (mu * c_query_term / self.collection_term_count))/(self.comment_term_count_dict[comment_offsets[comment_offsets_index]] + mu))
                    comment_offsets_index += 1
            while comment_offsets_index < len(comment_offsets):
                #no matches found
                score_list[comment_offsets_index] += math.log(((mu * c_query_term / self.collection_term_count))/(self.comment_term_count_dict[comment_offsets[comment_offsets_index]] + mu))
                comment_offsets_index += 1

        return score_list

    # load comment from given offset into comment file
    def load_comment(self, offset):
        self.comment_file.seek(offset)
        comment_as_list = next(self.comment_csv_reader)
        comment = Comment()
        comment.cid = int(comment_as_list[0])
        comment.url = comment_as_list[1]
        comment.author = comment_as_list[2]
        comment.time = comment_as_list[3]
        if comment_as_list[4] == 'None':
            comment.parent = None
        else:
            comment.parent = int(comment_as_list[4])
        comment.likes = int(comment_as_list[5])
        comment.dislikes = int(comment_as_list[6])
        comment.text = comment_as_list[7]
        return comment

    # binary search in seek list for term as key
    def get_index_in_seek_list(self, term):
        lb = 0
        rb = len(self.seek_list)
        while lb < rb:
            m = int(math.floor((lb + rb) / 2))
            comp_term = self.seek_list[m][0]
            if comp_term == term:
                return m
            elif comp_term < term:
                lb = m + 1
            else:
                rb = m
        return -1

    # return range of indices from first to last term which could start with prefix in seeklist
    def get_index_range_in_seek_list(self, prefix):
        # if stem starts with prefix the full word will as well
        # if prefix starts with stem the full word might start with prefix
        def maybe_prefix(stem):
            return stem.startswith(prefix) or prefix.startswith(stem)

        lb = 0
        rb = len(self.seek_list)
        first_found = None
        while lb < rb and first_found == None:
            m = int(math.floor((lb + rb) / 2))
            stem = self.seek_list[m][0]
            if maybe_prefix(stem):
                first_found = m
            elif stem < prefix:
                lb = m + 1
            else:
                rb = m

        if first_found == None:
            return range(0, 0)

        # could be done in O(logN) instead of O(N)
        lowest_index = first_found
        while lowest_index > 0 and maybe_prefix(self.seek_list[lowest_index-1][0]):
            lowest_index -= 1

        highest_index = first_found+1
        while highest_index < len(self.seek_list) and maybe_prefix(self.seek_list[highest_index][0]):
            highest_index += 1

        # inclusive lowest_index, exclusive highest_index
        return range(lowest_index, highest_index)

    # returns offsets into comment file for all comments containing stem in ascending order
    def get_offsets_for_stem(self, stem):
        i = self.get_index_in_seek_list(stem)
        if i == -1:
            return []
        self.index_file.seek(self.seek_list[i][1])
        posting_list = self.index_file.readline().rstrip('\n')
        posting_list_parts = posting_list.split(':')
        return [int(x.split(',')[0]) for x in posting_list_parts[2:]]

    # returns offsets into comment file for all comments containing stem in ascending order,
    # where either prefix starts with stem (false positive possible) or stem starts with prefix
    def get_offsets_for_prefix(self, prefix):
        index_range = self.get_index_range_in_seek_list(prefix)
        offsets_for_prefix = set() # prevent duplicate offsets
        for i in index_range:
            self.index_file.seek(self.seek_list[i][1])
            posting_list = self.index_file.readline().rstrip('\n')
            posting_list_parts = posting_list.split(':')
            offsets = [int(x.split(',')[0]) for x in posting_list_parts[2:]]
            for offset in offsets:
                offsets_for_prefix.add(offset)
        return offsets_for_prefix

    def get_comment_offsets_for_phrase_query(self, query):
        match = re.search(r'\'[^"]*\'', query)
        if not match:
            print('invalid phrase query')
            exit()
        phrase = match.group()[1:-1]
        new_query = ' AND '.join(phrase.split(' '))
        possible_matches = self.get_comment_offsets_for_query(new_query)
        return [x for x in possible_matches if phrase in self.load_comment(x).text.lower()]

    # returns offsets into comment file for all comments matching the query in ascending order
    def get_comment_offsets_for_query(self, query):
        if "'" in query:
            # can only search for whole query as one phrase
            assert(query[0] == "'" and query[-1] == "'")
            return self.get_comment_offsets_for_phrase_query(query)

        if ' NOT ' in query:
            split_query = query.split(' NOT ', 1)
            return self.search_boolean_NOT(split_query[0], split_query[1])
        if ' AND ' in query:
            split_query = query.split(' AND ', 1)
            return self.search_boolean_AND(split_query[0], split_query[1])
        if ' OR ' in query:
            split_query = query.split(' OR ', 1)
            return self.search_boolean_OR(split_query[0], split_query[1])

        #assume we are left with single term at this point
        assert(' ' not in query)
        prefix = query[:-1].lower() if query[-1] == '*' else None

        if(prefix == None):
            return self.get_offsets_for_stem(self.stemmer.stemWord(query.lower()))
        else:
            offsets_for_prefix = self.get_offsets_for_prefix(prefix)
            # filter false positives
            result = []
            for offset in offsets_for_prefix:
                comment = self.load_comment(offset)
                raw_tokens = nltk.word_tokenize(str(comment.text.lower(), encoding='utf-8'))
                for token in raw_tokens:
                    if token.startswith(prefix):
                        result.append(offset)
                        break
            return result



    def search_boolean_NOT(self, query1, query2):
        results = []
        q1_results = self.get_comment_offsets_for_query(query1)
        q2_results = self.get_comment_offsets_for_query(query2)
        i = 0
        j = 0
        # should be equivalent: results = [ result for result in q1_results if result not in q2_results ]
        while i < len(q1_results):
            if j == len(q2_results):
                results.append(q1_results[i])
                i += 1
            elif q1_results[i] < q2_results[j]:
                results.append(q1_results[i])
                i += 1
            elif q1_results[i] > q2_results[j]:
                j += 1
            else:
                i += 1
                j += 1
        return results

    def search_boolean_AND(self, query1, query2):
        results = []
        q1_results = self.get_comment_offsets_for_query(query1)
        q2_results = self.get_comment_offsets_for_query(query2)
        i = 0
        j = 0
        while i < len(q1_results) and j < len(q2_results):
            if q1_results[i] < q2_results[j]:
                i += 1
            elif q1_results[i] > q2_results[j]:
                j += 1
            else:
                results.append(q1_results[i])
                i += 1
                j += 1
        return results

    def search_boolean_OR(self, query1, query2):
        results = []
        q1_results = self.get_comment_offsets_for_query(query1)
        q2_results = self.get_comment_offsets_for_query(query2)
        i = 0
        j = 0
        while i < len(q1_results) or j < len(q2_results):
            if i == len(q1_results):
                results.append(q2_results[j])
                j += 1
            elif j == len(q2_results):
                results.append(q1_results[i])
                i += 1
            elif q1_results[i] < q2_results[j]:
                results.append(q1_results[i])
                i += 1
            elif q1_results[i] > q2_results[j]:
                results.append(q2_results[j])
                j += 1
            else:
                results.append(q1_results[i])
                i += 1
                j += 1
        return results

    def is_boolean_query(self, query):
        return ' AND ' in query or ' OR ' in query or ' NOT ' in query or '*' in query


    def search(self, query, top_k = 10):

        print(f'--------------------------------------------------searching for "{query}":')

        if self.is_boolean_query(query):
            comment_offsets = self.get_comment_offsets_for_query(query)
            print(f'{len(comment_offsets)} comments matched the query')
            if len(comment_offsets) > 0:
                print('example comment:')
                random_index = random.randrange(0, len(comment_offsets))
                example_comment = self.load_comment(comment_offsets[random_index])
                print(example_comment.text)
            print()
            return


        if not "'" in query:
            query = ' OR '.join(query.split(' '))
            query_terms = query.split(' ')
        else:
            assert(query[0] == "'" and query[-1] == "'")
            query_terms = [query]

        t_begin_searching = time.clock()
        comment_offsets = self.get_comment_offsets_for_query(query)
        print(f'{time.clock() - t_begin_searching} seconds for searching')

        if len(comment_offsets) == 0:
            print('no comments matched the query')
            return
        print(f'{len(comment_offsets)} comments matched the query')

        print('calculating scores...')
        t_begin_ranking = time.clock()
        top_k_rated_comments = [] # min heap of tuples (score, comment_offset)

        scores = self.get_dirichlet_smoothed_score(query_terms, comment_offsets)
        for i, comment_offset in enumerate(comment_offsets):
            score = scores[i]
            if len(top_k_rated_comments) < top_k:
                heapq.heappush(top_k_rated_comments, (score, comment_offset))
            else:
                heapq.heappushpop(top_k_rated_comments, (score, comment_offset))
        t_elapsed = time.clock() - t_begin_ranking
        print(f'{t_elapsed} seconds for scoring, {t_elapsed/len(comment_offsets)} per comment\n\n')

        print('results:')
        top_k_rated_comments.sort(key=lambda x: x[0], reverse=True)
        for score, comment_offset in top_k_rated_comments:
            print(f'score: {score}, text:')
            print(f'{self.load_comment(comment_offset).text}\n')


data_folder = 'data/real'
search_engine = SearchEngine()
# search_engine.index(data_folder)
# search_engine.compressIndex(data_folder)
search_engine.loadIndex(data_folder)
# search_engine.loadCompressedIndex(data_folder)
# print('index loaded')

# query = 'tragic west'
queries = ["tragic"]
for query in queries:
    search_engine.search(query, 5)
    print('\n\n\n')
