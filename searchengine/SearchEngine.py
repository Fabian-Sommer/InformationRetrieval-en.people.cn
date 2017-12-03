#!/usr/bin/python
from __future__ import division
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
        return data

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
        # self.stemmer = nltk.stem.PorterStemmer()

    def index(self, directory):
        #read csv
        comment_list = []
        with open(directory+"/comments.csv", 'rb') as f:
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

        print "Parsed csv into " + str(len(comment_list)) + " comments."

        #process comments (tokenize, remove stopwords, stem tokens)
        comments_processed = 0
        # phrase queries need stopwords...
        # stops = set(stopwords.words("english") + list(string.punctuation))
        for comment in comment_list:
            raw_tokens = nltk.word_tokenize(unicode(comment.text.lower(), 'utf-8'))
            comment.term_list = self.stemmer.stemWords(raw_tokens)
            comments_processed += 1
            if comments_processed % 1000 == 0:
                print str(comments_processed) + "/" + str(len(comment_list)) + " comments processed"
        print str(comments_processed) + "/" + str(len(comment_list)) + " comments processed - done"

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
            for stem, positions in comment_dict.iteritems():
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
        with open(directory+"/index.csv", 'wb') as f:
            for stem, posting_list in sorted_all_comment_dict.iteritems():
                line_string = ''
                line_string += '"' + stem.replace('"', '""').encode('utf-8') + '"'
                line_string += ':' + str(term_count_dict[stem])
                sorted_posting_list = [x for x in sorted(posting_list)]
                for posting_list_part in sorted_posting_list:
                    line_string += ':'
                    line_string += str(posting_list_part[0])
                    for position in posting_list_part[1]:
                        line_string += ','
                        line_string += str(position)
                line_string += '\n'
                f.write(line_string)
                offset_dict[stem] = current_offset
                current_offset += len(line_string)

        #seek list should be a sorted list
        seek_list = [(k, offset_dict[k]) for k in sorted(offset_dict)]

        #pickle out offset_dict
        with open(directory+"/seek_list.pickle", 'wb') as f:
            pickle.dump(seek_list, f, pickle.HIGHEST_PROTOCOL)

        with open(directory+"/comment_term_count_dict.pickle", 'wb') as f:
            pickle.dump(comment_term_count_dict, f, pickle.HIGHEST_PROTOCOL)

        with open(directory+"/collection_term_count.pickle", 'wb') as f:
            pickle.dump(collection_term_count, f, pickle.HIGHEST_PROTOCOL)

    def loadIndex(self, directory):
        with open(directory+"/seek_list.pickle", 'rb') as f:
            self.seek_list = pickle.load(f)
        with open(directory+"/comment_term_count_dict.pickle", 'rb') as f:
            self.comment_term_count_dict = pickle.load(f)
        with open(directory+"/collection_term_count.pickle", 'rb') as f:
            self.collection_term_count = pickle.load(f)
        self.comment_file = open(directory+"/comments.csv", 'rb')
        self.index_file = open(directory+"/index.csv", 'rb')
        self.comment_csv_reader = csv.reader(CSVInputFile(self.comment_file), quoting=csv.QUOTE_ALL)

    # returns score for ranking based on natural language model with dirichlet smoothing
    # query_terms: list of query terms, stemmed and filtered
    # comment_offset: offset of comment into comment file
    def get_dirichlet_smoothed_score(self, query_terms, comment_offset):
        mu = 1500
        score = 0
        terms_in_comment = self.comment_term_count_dict[comment_offset]

        for query_term in query_terms:
            i = self.get_index_in_seek_list(query_term)
            if i == -1:
                next
            self.index_file.seek(self.seek_list[i][1])
            posting_list = self.index_file.readline().rstrip('\n')
            posting_list_parts = posting_list.split(":")
            c_query_term = int(posting_list_parts[1])
            fD_query_term = 0
            for comment_list in posting_list_parts[2:]:
                #possible performance improvement: this can be sorted previously, then use bin. search
                occurences = comment_list.split(',')
                if int(occurences[0]) == comment_offset:
                    fD_query_term = len(occurences) - 1
                    break

            score += math.log((fD_query_term + (mu * c_query_term / self.collection_term_count))/(terms_in_comment + mu))

        return score
            

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
        posting_list_parts = posting_list.split(":")
        return [int(x.split(",")[0]) for x in posting_list_parts[2:]]

    # returns offsets into comment file for all comments containing stem in ascending order,
    # where either prefix starts with stem (false positive possible) or stem starts with prefix
    def get_offsets_for_prefix(self, prefix):
        index_range = self.get_index_range_in_seek_list(prefix)
        offsets_for_prefix = set() # prevent duplicate offsets
        for i in index_range:
            self.index_file.seek(self.seek_list[i][1])
            posting_list = self.index_file.readline().rstrip('\n')
            posting_list_parts = posting_list.split(":")
            offsets = [int(x.split(",")[0]) for x in posting_list_parts[2:]]
            for offset in offsets:
                offsets_for_prefix.add(offset)
        return offsets_for_prefix

    def get_comment_offsets_for_phrase_query(self, query):
        match = re.search(r'\'[^"]*\'', query)
        if not match:
            print "illegal phrase query"
            exit()
        phrase = match.group()[1:-1]
        new_query = " AND ".join(phrase.split(" "))
        possible_matches = self.get_comment_offsets_for_query(new_query)
        return [x for x in possible_matches if phrase in self.load_comment(x).text.lower()]

    # returns offsets into comment file for all comments matching the query in ascending order
    def get_comment_offsets_for_query(self, query):
        if "\'" in query:
            return self.get_comment_offsets_for_phrase_query(query)

        if " NOT " in query:
            split_query = query.split(" NOT ", 1)
            return self.search_boolean_NOT(split_query[0], split_query[1])
        if " AND " in query:
            split_query = query.split(" AND ", 1)
            return self.search_boolean_AND(split_query[0], split_query[1])
        if " OR " in query:
            split_query = query.split(" OR ", 1)
            return self.search_boolean_OR(split_query[0], split_query[1])

        #assume we are left with single term at this point
        assert(" " not in query)
        prefix = query[:-1].lower() if query[-1] == "*" else None

        if(prefix == None):
            return self.get_offsets_for_stem(self.stemmer.stemWord(query.lower()))
        else:
            offsets_for_prefix = self.get_offsets_for_prefix(prefix)
            # filter false positives
            result = []
            for offset in offsets_for_prefix:
                comment = self.load_comment(offset)
                raw_tokens = nltk.word_tokenize(unicode(comment.text.lower(), 'utf-8'))
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


    def search(self, query):
        print "--------------------------------------------------searching for \"" + query + "\":"
        comment_offsets = self.get_comment_offsets_for_query(query)
        print str(len(comment_offsets)) + " comments matched the query"
        if len(comment_offsets) > 0:
            print "example:"
            random_index = random.randrange(0, len(comment_offsets))
            example_comment = self.load_comment(comment_offsets[random_index])
            print example_comment.text
        print


data_folder = "data/real"
search_engine = SearchEngine()
#search_engine.index(data_folder)
search_engine.loadIndex(data_folder)
# query = "\'christmas market\'"
queries = [ "party AND chancellor", "party NOT politics", "war OR conflict", "euro* NOT europe", \
            "publi* OR moderation", "'the european union'", "'christmas market'" ]
#for query in queries:
    #search_engine.search(query)

print search_engine.get_dirichlet_smoothed_score(["Tragic"], 0)
