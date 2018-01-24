#!/usr/bin/env python3

import csv
import sys

from Common import CSVInputFile


def replace_newlines(csv_file_path):
    assert(csv_file_path[-4:] == ".csv")
    with open(csv_file_path, 'rb') as input_file, \
            open(f'{csv_file_path[:-4]}_without_newlines.csv', 'w') \
            as output_file:
        csv_reader = csv.reader(CSVInputFile(input_file),
                                quoting=csv.QUOTE_ALL)
        for csv_line in csv_reader:
            for i, item in enumerate(csv_line, start=1):
                item_without_newlines = item.replace("\n", " ")
                output_file.write(f'"{item_without_newlines}"'
                                  + ("," if i < len(csv_line) else "\n"))


def replace_chinese_punctuation(file_path):
    with open(file_path) as input_file, \
            open(f'{target_file}_without_chinese_punctuation', 'w') \
            as output_file:
        output_file.write(input_file.read().replace(
            '，', ', ').replace(
            '！', '! ').replace(
            '？', '? ').replace(
            '；', '; ').replace(
            '：', ': ').replace(
            '（', ' (').replace(
            '）', ') ').replace(
            '［', ' [').replace(
            '］', '] ').replace(
            '【', ' [').replace(
            '】', '] ').replace(
            '。', '. ')
        )


def read_line_generator(target_file):
    line = target_file.readline().rstrip('\n')
    while line:
        yield line
        line = target_file.readline().rstrip('\n')

# not used at the moment
# def set_cid_to_offset(comments_csv_path):
#     assert(comments_csv_path[-4:] == '.csv')
#     with open(comments_csv_path) as input_file, \
#             open(f'{comments_csv_path[:-4]}_offset_cids.csv', 'w') \
#             as output_file:
#         previous_offset = 0
#         for line in read_line_generator(input_file):
#             line_parts = line.partition(',')
#             output_file.write(f'"{previous_offset}",{line_parts[2]}\n')
#             previous_offset = input_file.tell()


if __name__ == '__main__':
    # replace_chinese_punctuation(sys.argv[1])
    # replace_newlines(sys.argv[1])
    # set_cid_to_offset(sys.argv[1])
