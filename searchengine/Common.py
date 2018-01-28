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


def read_line_generator(file_path):
    with open(file_path) as target_file:
        line = target_file.readline().rstrip('\n')
        while line:
            yield line
            line = target_file.readline().rstrip('\n')


def binary_read_line_generator_path(target_file_path):
    with open(target_file_path, mode='rb') as target_file:
        line = target_file.readline().decode().rstrip('\n')
        while line:
            yield line
            line = target_file.readline().decode().rstrip('\n')

def binary_read_line_generator(target_file):
    # use for files opened in binary mode
    # (maybe to first seek to offset and read from there)
    line = target_file.readline().decode().rstrip('\n')
    while line:
        yield line
        line = target_file.readline().decode().rstrip('\n')
