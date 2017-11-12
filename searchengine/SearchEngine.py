import csv
import nltk
from nltk.corpus import stopwords
import string
import pickle

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
        self.offset_dict = {}
        self.comment_file = None
        self.index_file = None
        self.comment_csv_reader = None
        self.stemmer = nltk.stem.PorterStemmer()

    def index(self, directory):
        #read csv
        comment_list = []
        with open(directory+"/comments2.csv", 'rb') as f:
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
                for posting_list_part in posting_list:
                    linestring += ':'
                    linestring += str(posting_list_part[0])
                    for position in posting_list_part[1]:
                        linestring += ','
                        linestring += str(position)
                linestring += '\n'
                file.write(linestring)
                offset_dict[stem] = current_offset
                current_offset += len(linestring)

        #pickle out offset_dict
        with open(directory+"/seek_list.pickle", 'wb') as file:
            pickle.dump(offset_dict, file, pickle.HIGHEST_PROTOCOL)





    def loadIndex(self, directory):
        with open(directory+"/seek_list.pickle", 'rb') as file:
            self.offset_dict = pickle.load(file)
        self.comment_file = open(directory+"/comments.csv", 'rb')
        self.index_file = open(directory+"/index.csv", 'rb')
        csvfile = CSVInputFile(self.comment_file)
        self.comment_csv_reader = csv.reader(self.comment_file, quoting=csv.QUOTE_ALL)

    def search(self, query):
        results = []

        #for now, query is single word
        query_term = self.stemmer.stem(query.lower())

        if query_term in self.offset_dict:
            index_offset = self.offset_dict[query_term]
            self.index_file.seek(index_offset)
            posting_list = self.index_file.readline().rstrip('\n')
            posting_list_parts = posting_list.split(":")
            result_offsets = []
            for i, posting_list_part in enumerate(posting_list_parts):
                if i == 0:
                    # this is just the query_term
                    continue
                if i > 5:
                    # we only want 5 results for now
                    continue
                entries = posting_list_part.split(",")
                result_offsets.append(int(entries[0])) # we dont need more information yet

            for comment_offset in result_offsets:
                self.comment_file.seek(comment_offset)
                #parse a single comment
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

                results.append(comment.text)

        return results

    def printAssignment2QueryResults(self):
        print searchEngine.search("October")
        print searchEngine.search("jobs")
        print searchEngine.search("Trump")
        print searchEngine.search("hate")

searchEngine = SearchEngine()
#searchEngine.index("E:/projects/InformationRetrieval/searchengine")
searchEngine.loadIndex("E:/projects/InformationRetrieval/searchengine")
searchEngine.printAssignment2QueryResults()