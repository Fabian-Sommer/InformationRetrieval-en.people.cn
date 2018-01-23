#!/usr/bin/python3

import Stemmer
import string
import pickle
import math
import random
import re
import heapq
import time
from sys import argv

from Report import Report
from Common import *
import Huffman


class SearchEngine():

    def __init__(self, using_compression=True):
        self.seek_list = []
        self.comment_file = None
        self.index_file = None
        self.huffman_tree_root = None
        self.comment_csv_reader = None
        self.comment_term_count_dict = None
        self.collection_term_count = 0
        self.stemmer = Stemmer.Stemmer('english')
        self.using_compression = using_compression
        self.report = Report()

    def load_index(self, directory):
        if self.using_compression:
            with open(f'{directory}/compressed_seek_list.pickle', mode='rb') \
                    as f:
                self.seek_list = pickle.load(f)
            self.index_file = open(f'{directory}/compressed_index', mode='rb')
            with open(f'{directory}/huffman_tree.pickle', mode='rb') as f:
                self.huffman_tree_root = pickle.load(f)
        else:
            with open(f'{directory}/seek_list.pickle', mode='rb') as f:
                self.seek_list = pickle.load(f)
            self.index_file = open(f'{directory}/index.csv',
                                   mode='r', encoding='utf-8')

        with open(f'{directory}/comment_term_count_dict.pickle', mode='rb') \
                as f:
            self.comment_term_count_dict = pickle.load(f)
        with open(f'{directory}/collection_term_count.pickle', mode='rb') as f:
            self.collection_term_count = pickle.load(f)
        self.comment_file = open(f'{directory}/comments.csv', mode='rb')
        self.comment_csv_reader = csv.reader(CSVInputFile(self.comment_file),
                                             quoting=csv.QUOTE_ALL)

    def load_posting_list_parts(self, stem):
        if self.using_compression:
            offset, size = self.seek_list[stem]
            self.index_file.seek(offset)
            binary_data = self.index_file.read(size)
            decoded_posting_list = Huffman.decode(
                binary_data, self.huffman_tree_root)
            return [stem] + decoded_posting_list.split(posting_list_separator)
        else:
            self.index_file.seek(self.seek_list[stem])
            posting_list = self.index_file.readline().rstrip('\n')
            return posting_list.split(posting_list_separator)

    # returns score based on natural language model with dirichlet smoothing
    # query_terms: list of query terms, stemmed and filtered
    # comment_offsets: list of offsets of comments into comment file
    def get_dirichlet_smoothed_score(self, query_stems, comment_offsets,
                                     mu=1500):
        score_list = [0 for x in comment_offsets]
        for query_stem in query_stems:
            if query_stem not in self.seek_list:
                continue
            posting_list_parts = self.load_posting_list_parts(query_stem)
            query_term_count = int(posting_list_parts[1])
            comment_offsets_index = 0
            for comment_list in posting_list_parts[2:]:
                if comment_offsets_index >= len(comment_offsets):
                    break
                first_occurence = int(comment_list.partition(',')[0])
                len_occurrences = comment_list.count(',') + 1
                while (comment_offsets_index < len(comment_offsets)
                        and first_occurence >
                        comment_offsets[comment_offsets_index]):
                    # term not found -> 0 occurences in comment
                    score_list[comment_offsets_index] += math.log(
                        (mu * query_term_count / self.collection_term_count)
                        / (self.comment_term_count_dict[comment_offsets[
                            comment_offsets_index]] + mu))
                    comment_offsets_index += 1

                if(comment_offsets_index < len(comment_offsets)
                        and first_occurence ==
                        comment_offsets[comment_offsets_index]):
                    fD_query_term = len_occurrences - 1
                    score_list[comment_offsets_index] += math.log(
                        (fD_query_term + (mu * query_term_count
                                          / self.collection_term_count))
                        / (self.comment_term_count_dict[comment_offsets[
                            comment_offsets_index]] + mu))
                    comment_offsets_index += 1
            while comment_offsets_index < len(comment_offsets):
                # no matches found
                score_list[comment_offsets_index] += math.log(
                    (mu * query_term_count / self.collection_term_count)
                    / (self.comment_term_count_dict[comment_offsets[
                        comment_offsets_index]] + mu))
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

    # returns offsets into comment file for all comments containing stem in
    # ascending order
    def get_offsets_for_stem(self, stem):
        if stem not in self.seek_list:
            return []
        posting_list_parts = self.load_posting_list_parts(stem)
        return [int(x.partition(',')[0]) for x in posting_list_parts[2:]]

    # returns offsets into comment file for all comments containing stem
    # starting with prefix
    def get_offsets_for_prefix(self, prefix):
        stems = self.seek_list.startswith(prefix)
        result = []
        for stem in stems:
            result += self.get_offsets_for_stem(stem)
        return result

    def get_comment_offsets_for_phrase_query(self, query):
        match = re.search(r'\'[^"]*\'', query)
        if not match:
            self.report.report('invalid phrase query')
            exit()
        phrase = match.group()[1:-1]
        new_query = phrase.replace(' ', ' AND ')
        possible_matches = self.get_comment_offsets_for_query(new_query)
        return [x for x in possible_matches
                if phrase in self.load_comment(x).text.lower()]

    # returns offsets into comment file for all comments matching the query in
    # ascending order
    def get_comment_offsets_for_query(self, query):
        if "'" in query:
            # can only search for whole query as one phrase
            assert(query[0] == "'" == query[-1])
            return self.get_comment_offsets_for_phrase_query(query)

        if ' NOT ' in query:
            split_query = query.partition(' NOT ')
            return self.search_boolean_NOT(split_query[0], split_query[2])
        if ' AND ' in query:
            split_query = query.partition(' AND ')
            return self.search_boolean_AND(split_query[0], split_query[2])
        if ' OR ' in query:
            split_query = query.partition(' OR ')
            return self.search_boolean_OR(split_query[0], split_query[2])

        # assume we are left with single term at this point
        assert(' ' not in query)

        if query[-1] == '*':
            return self.get_offsets_for_prefix(query[:-1].lower())
        else:
            return self.get_offsets_for_stem(
                self.stemmer.stemWord(query.lower()))

    # BOOLEAN QUERIES

    def search_boolean_NOT(self, query1, query2):
        results = []
        q1_results = self.get_comment_offsets_for_query(query1)
        q2_results = self.get_comment_offsets_for_query(query2)
        i = 0
        j = 0
        # should be equivalent:
        # results = [result for result in q1_results
        #            if result not in q2_results]
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
        return ' AND ' in query or ' OR ' in query or ' NOT ' in query \
            or '*' in query

    def search(self, query, top_k=10):

        self.report.report('-------------------------------------------------')
        self.report.report(f'searching for "{query}":')

        if self.is_boolean_query(query):
            comment_offsets = self.get_comment_offsets_for_query(query)
            self.report.report(
                f'{len(comment_offsets)} comments matched the query')
            if len(comment_offsets) > 0:
                self.report.report('example comment:')
                random_index = random.randrange(0, len(comment_offsets))
                example_comment = \
                    self.load_comment(comment_offsets[random_index])
                self.report.report(example_comment.text)
            self.report.report()
            return

        if "'" not in query:
            query = query.replace(' ', ' OR ')
            query_terms = [self.stemmer.stemWord(term.lower())
                           for term in query.split(' OR ')]
        else:
            assert(query[0] == "'" == query[-1])
            query_terms = [query]

        with self.report.measure('searching'):
            comment_offsets = self.get_comment_offsets_for_query(query)

        if len(comment_offsets) == 0:
            self.report.report('no comments matched the query')
            return
        self.report.report(
            f'{len(comment_offsets)} comments matched the query')

        with self.report.measure('calculating scores'):
            # min heap of tuples (score, comment_offset)
            top_k_rated_comments = []

            scores = self.get_dirichlet_smoothed_score(query_terms,
                                                       comment_offsets)
            for i, comment_offset in enumerate(comment_offsets):
                score = scores[i]
                if len(top_k_rated_comments) < top_k:
                    heapq.heappush(
                        top_k_rated_comments, (score, comment_offset))
                else:
                    heapq.heappushpop(top_k_rated_comments,
                                      (score, comment_offset))

        self.report.report('results:')
        top_k_rated_comments.sort(key=lambda x: x[0], reverse=True)
        for score, comment_offset in top_k_rated_comments:
            self.report.report(f'score: {score}, text:')
            self.report.report(f'{self.load_comment(comment_offset).text}\n')

        self.report.all_time_measures()


if __name__ == '__main__':
    data_directory = 'data/fake' if len(argv) < 2 else argv[1]
    search_engine = SearchEngine()
    search_engine.load_index(data_directory)
    search_engine.report.report('index loaded')

    queries = ["european"]
    for query in queries:
        search_engine.search(query, 5)
