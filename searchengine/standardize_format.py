#!/usr/bin/env python3

from sys import argv
import csv

def read_line_generator(file_path):
    with open(file_path) as target_file:
        line = target_file.readline().rstrip('\n')
        while line:
            yield line
            line = target_file.readline().rstrip('\n')


def standardize_format(csv_file_path, expected_number_of_fields=None):
    with open(f'{csv_file_path}.standardized.csv', 'w') as output_csv:
        csv_reader = csv.reader(read_line_generator(csv_file_path))
        csv_writer = csv.writer(output_csv)
        for i, row in enumerate(csv_reader, start=1):
            if expected_number_of_fields is not None \
                    and len(row) != expected_number_of_fields:
                print(f'unexpected number of fields: {len(row)} for this row:')
                print(row)
            csv_writer.writerow(
                (row[2], row[0], row[1], row[3], row[5], row[4], row[6]))

            if i % 100000 == 0:
                print(f'{i} rows standardized')



if __name__ == '__main__':
    assert(len(argv) == 2)
    standardize_format(argv[1], 7)
