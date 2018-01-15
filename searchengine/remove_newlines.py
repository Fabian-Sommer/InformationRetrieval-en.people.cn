#!/usr/bin/env python3

import csv
import sys

from Common import CSVInputFile

directory = sys.argv[1]

with open(f'{directory}/comments.csv', 'rb') as input_file, \
        open(f'{directory}/comments_without_newlines.csv', 'w') as output_file:
    csv_reader = csv.reader(CSVInputFile(input_file), quoting=csv.QUOTE_ALL)
    for csv_line in csv_reader:
        for i, item in enumerate(csv_line, start=1):
            item_without_newlines = item.replace("\n", " ")
            output_file.write(f'"{item_without_newlines}"'
                              + ("," if i < len(csv_line) else "\n"))
