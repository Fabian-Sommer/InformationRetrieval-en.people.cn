#!/usr/bin/env python3

import string
import pickle
import math
import random
import re
import heapq
import time
import functools
import sys
import numpy
from functools import reduce
from operator import itemgetter

import Stemmer
import nltk.tokenize
from dawg import RecordDAWG

from Report import Report
from Common import *
import Huffman
import QueryTree
from IRWS_Argument_Parsing import args


class SearchEngine():

    def __init__(self, using_compression=True):
        self.seek_list = None
        self.comment_file = None
        self.index_file = None
        self.symbol_to_encoding_dict = None
        self.cids = None
        self.comment_offsets_cid = None
        self.comment_offsets = None
        self.comment_term_counts = None
        self.comment_csv_reader = None
        self.authors_list = None
        self.articles_list = None
        self.reply_to_index = None
        self.collection_term_count = 0
        self.stemmer = Stemmer.Stemmer('english')
        self.tokenizer = nltk.tokenize.ToktokTokenizer()
        self.using_compression = using_compression
        self.report = Report()

    def load_index(self, directory):
        if self.using_compression:
            self.seek_list = RecordDAWG('>QQ')
            self.seek_list.load(f'{directory}/compressed_seek_list.dawg')
            self.index_file = open(f'{directory}/compressed_index', mode='rb')
            with open(f'{directory}/symbol_to_encoding_dict.pickle',
                      mode='rb') as f:
                self.symbol_to_encoding_dict = pickle.load(f)
        else:
            self.seek_list = RecordDAWG('>Q')
            self.seek_list.load(f'{directory}/seek_list.dawg')
            self.index_file = open(f'{directory}/index.csv',
                                   mode='r', encoding='utf-8')
        self.comment_offsets = numpy.load(
            f'{directory}/comment_offsets.npy', mmap_mode=None)
        self.comment_term_counts = numpy.load(
            f'{directory}/comment_term_counts.npy', mmap_mode=None)
        with open(f'{directory}/collection_term_count.pickle', mode='rb') as f:
            self.collection_term_count = pickle.load(f)
        self.comment_file = open(f'{directory}/comments.csv', mode='rb')
        self.comment_csv_reader = csv.reader(
            binary_read_line_generator(self.comment_file))
        with open(f'{directory}/authors_list.pickle', mode='rb') as f:
            self.authors_list = pickle.load(f)
        with open(f'{directory}/articles_list.pickle', mode='rb') as f:
            self.articles_list = pickle.load(f)
        with open(f'{directory}/reply_to_index.pickle', mode='rb') as f:
            self.reply_to_index = pickle.load(f)
        self.cids = numpy.load(f'{directory}/cids.npy', mmap_mode='r')
        self.comment_offsets_cid = numpy.load(
            f'{directory}/comment_offsets_cid.npy', mmap_mode='r')

    def load_collection_stem_count(self, stem):
        if self.using_compression:
            offset, size = self.seek_list[stem][0]
            self.index_file.seek(offset)
            binary_data = self.index_file.read(100)
            return Huffman.decode_first(
                binary_data, self.symbol_to_encoding_dict)
        else:
            self.index_file.seek(self.seek_list[stem][0])
            posting_list = self.index_file.readline().rstrip('\n')
            return int(posting_list.split(posting_list_separator,
                                          maxsplit=2)[1])

    def load_posting_list_parts(self, stem):
        if self.using_compression:
            offset, size = self.seek_list[stem][0]
            self.index_file.seek(offset)
            binary_data = self.index_file.read(size)
            decoded_posting_list = Huffman.decode(
                binary_data, self.symbol_to_encoding_dict)
            return [stem] + decoded_posting_list.split(posting_list_separator)
        else:
            self.index_file.seek(self.seek_list[stem][0])
            posting_list = self.index_file.readline().rstrip('\n')
            return posting_list.split(posting_list_separator)

    def get_comment_term_count(self, comment_offset):
        return self.comment_term_counts[numpy.searchsorted(
            self.comment_offsets, comment_offset)]

    def get_cid_to_offset(self, cid):
        return self.comment_offsets_cid[numpy.searchsorted(self.cids, cid)]

    # returns score based on natural language model with dirichlet smoothing
    # query_terms: list of query terms, stemmed and filtered
    # comment_offsets: list of offsets of comments into comment file
    def get_dirichlet_smoothed_score(self, query_terms, comment_offsets,
                                     mu=1500):
        ranked_comments = [[0, offset] for offset in comment_offsets]
        for query_term in query_terms:
            query_stem = self.stemmer.stemWord(query_term)
            if query_stem not in self.seek_list or \
                    self.seek_list[query_stem][0][1] > \
                    self.collection_term_count / 100:
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
                    ranked_comments[comment_offsets_index][0] += math.log(
                        (mu * query_term_count / self.collection_term_count)
                        / (self.get_comment_term_count(comment_offsets[
                            comment_offsets_index]) + mu))
                    comment_offsets_index += 1

                if(comment_offsets_index < len(comment_offsets)
                        and first_occurence ==
                        comment_offsets[comment_offsets_index]):
                    fD_query_term = len_occurrences - 1
                    ranked_comments[comment_offsets_index][0] += math.log(
                        (fD_query_term + (mu * query_term_count
                                          / self.collection_term_count))
                        / (self.get_comment_term_count(comment_offsets[
                            comment_offsets_index]) + mu))
                    comment_offsets_index += 1
            while comment_offsets_index < len(comment_offsets):
                # no matches found
                ranked_comments[comment_offsets_index][0] += math.log(
                    (mu * query_term_count / self.collection_term_count)
                    / (self.get_comment_term_count(comment_offsets[
                        comment_offsets_index]) + mu))
                comment_offsets_index += 1

        return ranked_comments

    # load comment from given offset into comment file
    def load_comment(self, offset):
        self.comment_file.seek(offset)
        comment_as_list = next(self.comment_csv_reader)
        comment = Comment()
        comment.cid = int(comment_as_list[0])
        # comment.article_url = self.articles_list[int(comment_as_list[1])]
        # comment.author = self.authors_list[int(comment_as_list[2])]
        comment.text = comment_as_list[3]
        # comment.timestamp = comment_as_list[4]
        # comment.parent_cid = int(comment_as_list[5]) \
        #    if comment_as_list[5] != '' else -1
        comment.upvotes = int(comment_as_list[6]) \
            if len(comment_as_list) >= 7 else 0
        comment.downvotes = int(comment_as_list[7]) \
            if len(comment_as_list) >= 8 else 0

        return comment

    def load_comment_from_cid(self, cid):
        return self.load_comment(self.get_cid_to_offset(cid))

    def load_cid_only(self, offset):
        self.comment_file.seek(offset)
        csv_line_start = self.comment_file.read(8)
        comma_position = csv_line_start.find(b',')
        while comma_position == -1:
            csv_line_start += self.comment_file.read(8)
            comma_position = csv_line_start.find(b',')
        return csv_line_start[:comma_position].decode()

    # returns offsets into comment file for all comments containing stem in
    # ascending order
    def get_offsets_for_stem(self, stem):
        if stem not in self.seek_list:
            return []
        posting_list_parts = self.load_posting_list_parts(stem)
        return [int(x.partition(',')[0]) for x in posting_list_parts[2:]]

    def get_comment_offsets_for_phrase_query(self, query):
        match = re.search(r'\'[^"]*\'', query)
        if not match:
            self.report.report('invalid phrase query')
            return []
        phrase = match.group()[1:-1]
        new_query = phrase.replace(' ', ' AND ')
        # reorder to have words with few matches first
        split_phrase = new_query.split(' AND ')
        occ_term_tuple_list = []
        for term in split_phrase:
            occ_term_tuple_list.append([self.load_collection_stem_count(
                self.stemmer.stemWord(term.lower())), term])
        occ_term_tuple_list.sort()
        # stopword removal
        term_list = map((lambda x: x[1]), list(filter((
            lambda x:
            x[0] < self.collection_term_count/100), occ_term_tuple_list)))
        if term_list == []:
            term_list = [occ_term_tuple_list[0][1]]
        new_query = reduce((lambda x, y: x + ' AND ' + y), term_list)
        possible_matches = self.get_comment_offsets_for_query(new_query)
        return [offset for offset in possible_matches
                if phrase in self.load_comment(offset).text.lower()]

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
            stems = self.seek_list.keys(query[:-1].lower())
            result = []
            for stem in stems:
                result.extend(self.get_offsets_for_stem(stem))
            return result
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
        return ' AND ' in query or ' OR ' in query or 'NOT ' in query

    def phrase_query(self, phrase, suffix=''):
        if ' ' not in phrase and suffix == '':
            return self.basic_search(phrase)

        stem_offset_size_list = []  # may contain duplicates!
        for sentence in nltk.tokenize.sent_tokenize(phrase):
            for token in self.tokenizer.tokenize(sentence):
                stem = self.stemmer.stemWord(token)
                if stem not in self.seek_list:
                    continue
                stem_offset_size_list.append((stem, self.seek_list[stem]))

        # sort by posting_list size
        stem_offset_size_list.sort(key=lambda t: t[1][0][1])
        smallest_stem = stem_offset_size_list[0][0]
        second_smallest_stem = stem_offset_size_list[1][0] \
            if len(stem_offset_size_list) > 1 and \
            stem_offset_size_list[1][1][0][1] < \
            self.collection_term_count / 100 else ''
        result = []
        phrase_to_check = phrase if suffix == '' else f'{phrase} {suffix}'
        offsets = set(self.get_offsets_for_stem(smallest_stem))
        if second_smallest_stem != '':
            offsets.intersection_update(
                self.get_offsets_for_stem(second_smallest_stem))
        for offset in offsets:
            comment = self.load_comment(offset)
            if phrase_to_check in comment.text.lower():
                result.append(offset)
        return result

    def basic_search(self, token_node):
        # search for a single query token

        if token_node.kind == 'phrase_prefix':  # phrase prefix query: 'hi ye'*
            return self.phrase_query(
                token_node.phrase_start, token_node.prefix)
        elif token_node.kind == 'phrase':  # phrase query: 'european union'
            return self.phrase_query(token_node.phrase)
        elif token_node.kind == 'prefix':  # prefix query: isra*
            stems_with_prefix = self.seek_list.keys(token_node.prefix)
            result = []
            for stem in stems_with_prefix:
                result.extend(self.get_offsets_for_stem(stem))
            return result
        elif token_node.kind == 'reply_to':  # ReplyTo query: ReplyTo:12345
            if token_node.target_cid not in self.reply_to_index.keys():
                return []

            return [self.cid_to_offset[cid]
                    for cid in self.reply_to_index[token_node.target_cid]]
        elif token_node.kind == 'keyword':  # keyword query: merkel
            return self.get_offsets_for_stem(
                self.stemmer.stemWord(token_node.keyword))
        else:
            raise RuntimeError(f'unknown token_node.kind: {token_node.kind}')

    def search(self, query, top_k=3, printIdsOnly=True):
        def show_comments(comment_iterable):
            comment_iterable = list(comment_iterable)
            print(len(comment_iterable))
            return
            if printIdsOnly:
                cids = (str(comment.cid) for comment in comment_iterable)
                self.report.report(','.join(cids))
            else:
                for comment in comment_iterable:
                    self.report.report(
                        f'id: {comment.cid}, text:\n{comment.text}\n')
            self.report.report()

        self.report.report(f'\nsearching for "{query}":')

        comment_offsets = []

        # boolean query
        if self.is_boolean_query(query):
            with self.report.measure('searching'):
                comment_offsets = self.get_comment_offsets_for_query(query)
            self.report.report(
                f'{len(comment_offsets)} comments matched the query')
            show_comments(map(self.load_comment, comment_offsets))
            return

        # ReplyTo query
        if query.startswith("ReplyTo:"):
            target_cid = int(query.partition("ReplyTo:")[2])
            if target_cid not in self.reply_to_index.keys():
                self.report.report(f'comment with id {target_cid} does not '
                                   'exist or it has no replies')
                return
            self.report.report("target comment:")
            show_comments((self.load_comment_from_cid(target_cid),))
            replies = []
            with self.report.measure('searching'):
                replies = self.reply_to_index[target_cid]
            self.report.report(f'found {len(replies)} replies:')
            show_comments(map(self.load_comment_from_cid, replies[:top_k]))
            return

        if "'" in query:
            # phrase query
            assert(query[0] == "'" == query[-1])
            query_terms = [query.lower()]
        else:
            # keyword query
            query = query.replace(' ', ' OR ')
            query_terms = [self.stemmer.stemWord(term.lower())
                           for term in query.split(' OR ')]

        with self.report.measure('searching'):
            comment_offsets = self.get_comment_offsets_for_query(query)

        self.report.report(
            f'{len(comment_offsets)} comments matched the query')

        if len(comment_offsets) == 0:
            return

        with self.report.measure('calculating scores'):
            # rated_comment is a tuple of (score, offset)
            rated_comments = self.get_dirichlet_smoothed_score(
                query_terms, comment_offsets)
            if top_k is not None and len(rated_comments) > top_k:
                top_k_rated_comments = rated_comments[:top_k]
                heapq.heapify(top_k_rated_comments)
                for rated_comment in rated_comments[top_k:]:
                    heapq.heappushpop(top_k_rated_comments, rated_comment)
                result = top_k_rated_comments
            else:
                result = rated_comments

            result.sort(key=lambda x: x[0], reverse=True)

        # show_comments(self.load_comment(comment_offset)
        #               for score, comment_offset in rated_comments)
        print(len(result))

    def search_new(self, query, top_k=None, printIdsOnly=True):
        self.report.report(f'\nsearching for "{query}":')

        def print_comments(offset_iterable):
            offset_iterable = list(offset_iterable)
            if printIdsOnly:
                print(','.join((self.load_cid_only(offset)
                                for offset in offset_iterable)))
            else:
                for offset in offset_iterable:
                    comment = self.load_comment(offset)
                    print(f'{comment.cid},{comment.text}')

        query_tree_root = QueryTree.build(query)
        if query_tree_root.is_boolean_query:
            or_result = set()
            with self.report.measure('searching'):
                for and_node in query_tree_root.children:
                    and_result = None
                    to_be_removed = []
                    for child in and_node.children:
                        child_result = self.basic_search(child)
                        if child.is_negated:
                            to_be_removed.append(child_result)
                        elif and_result is None:
                            and_result = set(child_result)
                        else:
                            and_result.intersection_update(child_result)
                    and_result.difference_update(*to_be_removed)
                    or_result.update(and_result)

            print_comments(or_result)
        else:  # non bool query
            with self.report.measure('searching'):
                children_results = (self.basic_search(child)
                                    for child in query_tree_root.children)
                comment_offsets = list(frozenset().union(*children_results))

            with self.report.measure('calculating scores'):
                # rated_comment is a tuple of (score, offset)
                rated_comments = self.get_dirichlet_smoothed_score(
                    query_tree_root.query_terms, comment_offsets)
                if top_k is not None and len(rated_comments) > top_k:
                    top_k_rated_comments = \
                        rated_comments[:top_k]
                    heapq.heapify(top_k_rated_comments)
                    for rated_comment in rated_comments[top_k:]:
                        heapq.heappushpop(top_k_rated_comments, rated_comment)
                    result = top_k_rated_comments
                else:
                    result = rated_comments

                result.sort(key=lambda x: x[0], reverse=True)

            print_comments((offset for score, offset in result))


if __name__ == '__main__':
    data_directory = 'data/guardian'  # TODO change to '.' before submitting
    search_engine = SearchEngine()
    search_engine.load_index(data_directory)
    search_engine.report.report('index loaded')

    for query in open(args.query):
        query = query.strip()
        # TODO remove
        if query.startswith('#'):
            continue
        if query.startswith('!'):
            query = query.partition('!')[2].strip()
            with search_engine.report.measure('old search'):
                search_engine.search(query, args.topN, args.printIdsOnly)
        with search_engine.report.measure('new search'):
            search_engine.search_new(query, args.topN, args.printIdsOnly)
        print('\n\n')
