import csv

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
