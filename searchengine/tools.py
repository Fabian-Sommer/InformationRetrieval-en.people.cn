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


def replace_chinese_punctuation(target_file):
    with open(target_file, 'r') as input_file, \
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


if __name__ == '__main__':
    # replace_chinese_punctuation(sys.argv[1])
    # replace_newlines(sys.argv[1])
