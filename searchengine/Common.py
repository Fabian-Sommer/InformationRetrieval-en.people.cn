import csv

posting_list_separator = '\a'  # 


class Comment():

    def __init__(self, cid=-1, article_url='', author='', text='',
                 timestamp='', parent_cid=-1, upvotes=-1, downvotes=-1,
                 file_offset=-1, term_list=None):
        self.cid = cid
        self.article_url = article_url
        self.author = author
        self.text = text
        self.timestamp = timestamp
        self.parent_cid = parent_cid
        self.upvotes = upvotes
        self.downvotes = downvotes
        self.file_offset = file_offset
        self.term_list = [] if term_list is None else term_list

    def __eq__(self, other):
        if self.cid == other.cid:
            if not (self.article_url == other.article_url and
                    self.author == other.author and
                    self.timestamp == other.timestamp and
                    self.parent_cid == other.parent_cid and
                    self.upvotes == other.upvotes and
                    self.downvotes == other.downvotes and
                    self.text == other.text and
                    self.file_offset == other.file_offset and
                    self.term_list == other.term_list):
                print('warning: same id but not equal:\n',
                      self, '\n!=\n', other)
                return False
            return True
        else:
            return False

    def __repr__(self):
        return f"Comment({self.cid}, '{self.article_url}', '{self.author}', " \
            f"'{self.text}', '{self.timestamp}', {self.parent_cid}, " \
            f"{self.upvotes}, {self.downvotes}, {self.file_offset}, " \
            f"{self.term_list})"

    def __hash__(self):
        return self.cid


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
