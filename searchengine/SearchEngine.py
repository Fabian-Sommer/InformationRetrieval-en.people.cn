#!/usr/bin/env python3

import string
import pickle
import math
import random
import re
import heapq
import time
import functools
from functools import reduce

import Stemmer
import nltk.tokenize
from dawg import RecordDAWG

from Report import Report
from Common import *
import Huffman
from IRWS_Argument_Parsing import args


class SearchEngine():

    def __init__(self, using_compression=True):
        self.seek_list = None
        self.comment_file = None
        self.index_file = None
        self.symbol_to_encoding_dict = None
        self.comment_csv_reader = None
        self.comment_term_count_dict = None
        self.authors_list = None
        self.articles_list = None
        self.reply_to_index = None
        self.cid_to_offset = None
        self.collection_term_count = 0
        self.stemmer = Stemmer.Stemmer('english')
        self.tokenizer = nltk.tokenize.ToktokTokenizer()
        self.using_compression = using_compression
        self.report = Report()

    def load_index(self, directory):
        if self.using_compression:
            self.seek_list = RecordDAWG('>II')
            self.seek_list.load(f'{directory}/compressed_seek_list.dawg')
            self.index_file = open(f'{directory}/compressed_index', mode='rb')
            with open(f'{directory}/symbol_to_encoding_dict.pickle',
                      mode='rb') as f:
                self.symbol_to_encoding_dict = pickle.load(f)
        else:
            self.seek_list = RecordDAWG('>I')
            self.seek_list.load(f'{directory}/seek_list.dawg')
            self.index_file = open(f'{directory}/index.csv',
                                   mode='r', encoding='utf-8')

        with open(f'{directory}/comment_term_count_dict.pickle',
                  mode='rb') as f:
            self.comment_term_count_dict = pickle.load(f)
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
        with open(f'{directory}/cid_to_offset.pickle', mode='rb') as f:
            self.cid_to_offset = pickle.load(f)

    def load_collection_stem_count(self, stem):
        if self.using_compression:
            offset, size = self.seek_list[stem][0]
            self.index_file.seek(offset)
            binary_data = self.index_file.read(100)
            return Huffman.decode_first(binary_data, self.symbol_to_encoding_dict)
        else:
            self.index_file.seek(self.seek_list[stem][0])
            posting_list = self.index_file.readline().rstrip('\n')
            return int(posting_list.split(posting_list_separator, maxsplit=2)[1])

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
        comment.article_url = self.articles_list[int(comment_as_list[1])]
        comment.author = self.authors_list[int(comment_as_list[2])]
        comment.text = comment_as_list[3]
        comment.timestamp = comment_as_list[4]
        comment.parent_cid = int(comment_as_list[5]) \
            if comment_as_list[5] != '' else -1
        comment.upvotes = int(comment_as_list[6]) \
            if len(comment_as_list) >= 7 else 0
        comment.downvotes = int(comment_as_list[7]) \
            if len(comment_as_list) >= 8 else 0

        return comment

    def load_comment_from_cid(self, cid):
        return self.load_comment(self.cid_to_offset[cid])


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
            occ_term_tuple_list.append([self.load_collection_stem_count(self.stemmer.stemWord(term.lower())), term])
        occ_term_tuple_list.sort()
        #stopword removal
        term_list = map((lambda x: x[1]), list(filter((lambda x: x[0] < self.collection_term_count/100),occ_term_tuple_list))) 
        if term_list == []:
            term_list = [occ_term_tuple_list[0][1]]
        new_query = reduce((lambda x, y: x + ' AND ' + y), term_list)
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
        return ' AND ' in query or ' OR ' in query or ' NOT ' in query

    def basic_search(self, query):
        # search for a single query token
        # TODO call this after properly splitting up query

        if len(query) == 0:
            return []

        query = query.lower()

        # phrase prefix query: 'new ye'*
        if len(query) > 1 and query[-2] == "'":
            # TODO this works but is toooo slow
            assert(query[0] == "'" and query.count("'") == 2
                   and query[-1] == '*')

            parts = query[1:-2].rpartition(' ')
            phrase_start = parts[0]
            prefix = parts[2]
            phrases = (f"'{phrase_start} {stem}'"
                       for stem in self.seek_list.keys(prefix))
            result = set()
            for phrase in phrases:
                if phrase.count("'") > 2:
                    print(phrase)
                    print(phrase_start)
                    print(prefix)
                for offset in self.basic_search(phrase):
                    result.add(offset)
            return list(result)

        # phrase query: 'european union'
        elif query[-1] == "'":
            assert(query[0] == "'" and query.count("'") == 2)

            stem_offset_size_list = []  # may contain duplicates!
            phrase = query[1:-1]
            for sentence in nltk.tokenize.sent_tokenize(phrase):
                for token in self.tokenizer.tokenize(sentence):
                    stem = self.stemmer.stemWord(token)
                    if stem not in self.seek_list:
                        continue
                    stem_offset_size_list.append((stem, self.seek_list[stem]))

            # sort by posting_list size
            stem_offset_size_list.sort(key=lambda t: t[1][0][1])
            smallest_stem = stem_offset_size_list[0][0]
            result = []
            for offset in self.get_offsets_for_stem(smallest_stem):
                comment = self.load_comment(offset)
                if phrase in comment.text:
                    result.append(offset)
            return result

        # prefix query: isra*
        elif query[-1] == '*':
            assert(query.count('*') == 1)

            prefix = query[:-1]
            stems_with_prefix = self.seek_list.keys(prefix)
            result = []
            for stem in stems_with_prefix:
                result.extend(self.get_offsets_for_stem(stem))
            return result

        # ReplyTo query: ReplyTo:12345
        elif 'ReplyTo:' in query:
            assert(query.count('ReplyTo:') == 1 and
                   query.startswith('ReplyTo:'))
            target_cid = int(query.partition('ReplyTo:')[2])
            if target_cid not in self.reply_to_index.keys():
                return []
            return self.reply_to_index[target_cid]

        # keyword query: merkel
        else:
            assert(' ' not in query)
            return self.get_offsets_for_stem(self.stemmer.stemWord(query))

    def search(self, query, top_k=3, printIdsOnly=False):
        def show_comments(comment_iterable):
            if printIdsOnly:
                cids = (str(comment.cid) for comment in comment_iterable)
                self.report.report(','.join(cids))
            else:
                for comment in comment_iterable:
                    self.report.report(
                        f'id: {comment.cid}, text:\n{comment.text}\n')
            self.report.report()

        self.report.report('-------------------------------------------------')
        self.report.report(f'searching for "{query}":')

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

        top_k_rated_comments.sort(key=lambda x: x[0], reverse=True)
        show_comments(self.load_comment(comment_offset)
                      for score, comment_offset in top_k_rated_comments)


if __name__ == '__main__':
    data_directory = 'data/people'  # TODO change to '.' before submitting
    search_engine = SearchEngine()
    search_engine.load_index(data_directory)
    search_engine.report.report('index loaded')

    topN = 3 if args.topN is None else args.topN

    for query in read_line_generator(args.query):
        if query.startswith('#'):  # TODO remove
            continue
        search_engine.search(query, topN, args.printIdsOnly)
        search_engine.report.all_time_measures()
        print('\n\n')

    # print(search_engine.seek_list.keys('unreasonabl'))
    # print(search_engine.seek_list['ye*'])

    # for offset in search_engine.basic_search("'nothing unreasonabl'*"):
    #     comment = search_engine.load_comment(offset)
    #     print(comment.text)
