#!/usr/bin/env python3

import csv
import sys

from Common import binary_read_line_generator, read_line_generator


def replace_newlines(csv_file_path):
    assert(csv_file_path[-4:] == ".csv")
    with open(csv_file_path, mode='rb') as input_file, \
            open(f'{csv_file_path[:-4]}_without_newlines.csv', mode='w') \
            as output_file:
        csv_reader = csv.reader(binary_read_line_generator(input_file))
        for csv_line in csv_reader:
            for i, item in enumerate(csv_line, start=1):
                item_without_newlines = item.replace("\n", " ")
                output_file.write(f'"{item_without_newlines}"'
                                  + ("," if i < len(csv_line) else "\n"))


def replace_chinese_punctuation(file_path):
    with open(file_path) as input_file, \
            open(f'{target_file}_without_chinese_punctuation', mode='w') \
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


def fix_csv(csv_file_path):
    with open(csv_file_path, newline='') as csv_file, \
            open(f'{csv_file_path}.fixed', mode='w', newline='') as csv_output_file:
        csv_writer = csv.writer(csv_output_file)
        for line in read_line_generator(csv_file):
            csv_writer.writerow(line[1:-1].split('","'))


def check_comment_parsing(csv_file_path, number_of_fields=8):
    number_of_violating_comments = 0
    total = 0
    with open(csv_file_path, newline='') as csv_file:
        for csv_line in csv.reader(read_line_generator(csv_file)):
            total += 1
            if len(csv_line) != number_of_fields:
                if number_of_violating_comments < 3:
                    print(csv_line, '\n\n')
                number_of_violating_comments += 1
    print(f'{number_of_violating_comments}/{total} comments have an',
          'unexpected number of fields')


# not used at the moment
# def set_cid_to_offset(comments_csv_path):
#     assert(comments_csv_path[-4:] == '.csv')
#     with open(comments_csv_path) as input_file, \
#             open(f'{comments_csv_path[:-4]}_offset_cids.csv', mode='w') \
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
    # fix_csv(sys.argv[1])
    check_comment_parsing(sys.argv[1])
