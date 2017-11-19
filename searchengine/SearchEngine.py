import csv
import nltk
from nltk.corpus import stopwords
import string
import pickle
import math

class Comment():
    cid = 0
    url = ''
    author = ''
    time = ''
    parent = 0
    likes = 0
    dislikes = 0
    text = ''
    fileoffset = 0
    token_list = []

class CSVInputFile(object):
    """ File-like object. """
    def __init__(self, file):
        self.file = file
        self.offset = None
        self.linelen = None

    def __iter__(self):
        return self

    def __next__(self):
        offset = self.file.tell()
        data = self.file.readline()
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
        self.stemmer = nltk.stem.PorterStemmer()

    def index(self, directory):
        #read csv
        comment_list = []
        with open(directory+"/comments.csv", 'rb') as f:
            csvfile = CSVInputFile(f)
            reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
            lastoffset = 0
            for row in reader:
                comment = Comment()
                comment.cid = int(row[0])
                comment.url = row[1]
                comment.author = row[2]
                comment.time = row[3]
                if row[4] == 'None':
                    comment.parent = None
                else:
                    comment.parent = int(row[4])
                comment.likes = int(row[5])
                comment.dislikes = int(row[6])
                comment.text = row[7]
                comment.fileoffset = lastoffset
                lastoffset = f.tell()
                comment_list.append(comment)
        print "Parsed csv into " + str(len(comment_list)) + " comments."

        #process comments (tokenize, remove stopwords, stem tokens)
        prog = 0
        stops = set(stopwords.words("english") + list(string.punctuation))
        for comment in comment_list:
            comment.token_list = [self.stemmer.stem(word) for word in nltk.word_tokenize(unicode(comment.text.lower(), 'utf-8')) if word not in stops]
            prog += 1
            if prog % 1000 == 0:
                print prog / 1000
        
        #create index
        all_comment_dict = {}
        for comment in comment_list:
            position = 0
            comment_dict = {}
            for stem in comment.token_list:
                if not stem in comment_dict:
                    comment_dict[stem] = []
                comment_dict[stem].append(position)
                position += 1
            for stem, positions in comment_dict.iteritems():
                if not stem in all_comment_dict:
                    all_comment_dict[stem] = []
                all_comment_dict[stem].append([comment.fileoffset, positions])
        
        #save index as csv
        offset_dict = {}
        current_offset = 0
        with open(directory+"/index.csv", 'wb') as file:
            for stem, posting_list in all_comment_dict.iteritems():
                linestring = ''
                linestring += '"' + stem.replace('"', '""').encode('utf-8') + '"'
                sorted_posting_list = [x for x in sorted(posting_list)]
                for posting_list_part in sorted_posting_list:
                    linestring += ':'
                    linestring += str(posting_list_part[0])
                    for position in posting_list_part[1]:
                        linestring += ','
                        linestring += str(position)
                linestring += '\n'
                file.write(linestring)
                offset_dict[stem] = current_offset
                current_offset += len(linestring)

        #seek list should be a sorted list
        seek_list = [(k, offset_dict[k]) for k in sorted(offset_dict)]
        #pickle out offset_dict
        with open(directory+"/seek_list.pickle", 'wb') as file:
            pickle.dump(seek_list, file, pickle.HIGHEST_PROTOCOL)

    def loadIndex(self, directory):
        with open(directory+"/seek_list.pickle", 'rb') as file:
            self.seek_list = pickle.load(file)
        self.comment_file = open(directory+"/comments.csv", 'rb')
        self.index_file = open(directory+"/index.csv", 'rb')
        csvfile = CSVInputFile(self.comment_file)
        self.comment_csv_reader = csv.reader(self.comment_file, quoting=csv.QUOTE_ALL)

    # load comment from given offset into comment file
    def loadComment(self, offset):
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
            if (comp_term == term):
                return m
            elif comp_term < term:
                lb = m + 1
            else:
                rb = m
        return -1

    # returns offsets into comment file for all comments containing stem in ascending order
    def getOffsetsForStem(self, stem):
        i = self.get_index_in_seek_list(stem)
        if i == -1: 
            return []
        self.index_file.seek(self.seek_list[i][1])
        posting_list = self.index_file.readline().rstrip('\n')
        posting_list_parts = posting_list.split(":")
        return [int(x.split(",")[0]) for x in posting_list_parts[1:]]

    # returns offsets into comment file for all comments matching the query in ascending order
    def getCommentOffsetsForQuery(self, query):
        #only single term and BOOLEANS implemented
        #TODO: handle *, phrase query
        if " NOT " in query:
            split_query = query.split(" NOT ", 1)
            return self.searchBooleanNOT(split_query[0], split_query[1])
        if " AND " in query:
            split_query = query.split(" AND ", 1)
            return self.searchBooleanAND(split_query[0], split_query[1])
        if " OR " in query:
            split_query = query.split(" OR ", 1)
            return self.searchBooleanOR(split_query[0], split_query[1])

        #TODO: check for phrase query or * here
        
        #assume we are left with single term at this point
        return self.getOffsetsForStem(self.stemmer.stem(query.lower()))

    def searchBooleanNOT(self, query1, query2):
        results = []
        q1_results = self.getCommentOffsetsForQuery(query1)
        q2_results = self.getCommentOffsetsForQuery(query2)
        i = 0
        j = 0
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

    def searchBooleanAND(self, query1, query2):
        results = []
        q1_results = self.getCommentOffsetsForQuery(query1)
        q2_results = self.getCommentOffsetsForQuery(query2)
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

    def searchBooleanOR(self, query1, query2):
        results = []
        q1_results = self.getCommentOffsetsForQuery(query1)
        q2_results = self.getCommentOffsetsForQuery(query2)
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
        comment_offsets = self.getCommentOffsetsForQuery(query)
        print len(comment_offsets)
        if len(comment_offsets) > 0:
            example_comment = self.loadComment(comment_offsets[0])
            print example_comment.text


searchEngine = SearchEngine()
#searchEngine.index("E:/projects/InformationRetrieval/searchengine")
searchEngine.loadIndex("E:/projects/InformationRetrieval/searchengine")
searchEngine.search("party AND chancellor")