import csv


class Comment():

    def __init__(self, cid=0, url='', author='', time='', parent=0, likes=0,
                 dislikes=0, text='', file_offset=0):
        self.cid = cid
        self.url = url
        self.author = author
        self.time = time
        self.parent = parent
        self.likes = likes
        self.dislikes = dislikes
        self.text = text
        self.file_offset = file_offset
        self.term_list = []

    def init_from_csv_line(self, csv_line, file_offset):
        self.cid = int(csv_line[0])
        self.url = csv_line[1]
        self.author = csv_line[2]
        self.time = csv_line[3]
        self.parent = None if csv_line[4] == 'None' else int(csv_line[4])
        self.likes = int(csv_line[5])
        self.dislikes = int(csv_line[6])
        self.text = csv_line[7]
        self.file_offset = file_offset
        return self

    def __eq__(self, other):
        return self.cid == other.cid and \
            self.url == other.url and \
            self.author == other.author and \
            self.time == other.time and \
            self.parent == other.parent and \
            self.likes == other.likes and \
            self.dislikes == other.dislikes and \
            self.text == other.text and \
            self.file_offset == other.file_offset and \
            self.term_list == other.term_list

    def __repr__(self):
        return f"Comment({self.cid}, '{self.url}', '{self.author}', "
        "'{self.time}', {self.parent}, {self.likes}, {self.dislikes}, "
        "'{self.text}', {self.file_offset}, {self.term_list})"


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
