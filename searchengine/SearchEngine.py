#!/usr/bin/python3

import Stemmer
import string
import pickle
import math
import random
import re
import heapq
import time

from Common import *
from Huffman import decode_huffman

class SearchEngine():

    def __init__(self):
        self.seek_list = []
        self.comment_file = None
        self.index_file = None
        self.comment_csv_reader = None
        self.comment_term_count_dict = None
        self.collection_term_count = 0
        self.stemmer = Stemmer.Stemmer('english')

    def loadIndex(self, directory, use_compressed = True):
        if use_compressed:
            with open(f'{directory}/compressed_seek_list.pickle', mode='rb') as f:
                self.seek_list = pickle.load(f)
            self.index_file = open(f'{directory}/compressed_index', mode='rb')
        else:
            with open(f'{directory}/seek_list.pickle', mode='rb') as f:
                self.seek_list = pickle.load(f)
            self.index_file = open(f'{directory}/index.csv', mode='r', encoding='utf-8')

        with open(f'{directory}/comment_term_count_dict.pickle', mode='rb') as f:
            self.comment_term_count_dict = pickle.load(f)
        with open(f'{directory}/collection_term_count.pickle', mode='rb') as f:
            self.collection_term_count = pickle.load(f)
        self.comment_file = open(f'{directory}/comments.csv', mode='rb')
        self.comment_csv_reader = csv.reader(CSVInputFile(self.comment_file), quoting=csv.QUOTE_ALL)

    # returns score for ranking based on natural language model with dirichlet smoothing
    # query_terms: list of query terms, stemmed and filtered
    # comment_offsets: list of offsets of comments into comment file
    def get_dirichlet_smoothed_score(self, query_terms, comment_offsets, mu = 1500):
        score_list = [ 0 for x in comment_offsets ]
        for query_term in query_terms:
            self.index_file.seek(self.seek_list[query_term])
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
        self.index_file.seek(self.seek_list[stem])
        posting_list = self.index_file.readline().rstrip('\n')
        posting_list_parts = posting_list.split(':')
        return [ int(x.split(',')[0]) for x in posting_list_parts[2:] ]

    # returns offsets into comment file for all comments containing stem starting with prefix
    def get_offsets_for_prefix(self, prefix):
        stems = self.seek_list.startswith(prefix)
        result = []
        for stem in stems:
            result += self.get_offsets_for_stem(stem)
        return result

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
            return self.get_offsets_for_prefix(prefix)

    # BOOLEAN QUERIES

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

if __name__ == '__main__':
    data_folder = 'data/fake'
    search_engine = SearchEngine()
    search_engine.loadIndex(data_folder, use_compressed = False)
    # search_engine.loadCompressedIndex(data_folder)
    # print('index loaded')

    queries = ["inte*"]
    for query in queries:
        search_engine.search(query, 5)
        print('\n\n\n')
