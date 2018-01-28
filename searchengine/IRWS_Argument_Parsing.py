#!/usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("query", help="a txt file with one boolean, keyword,"
                    "phrase, ReplyTo, or Index query per line")
parser.add_argument("--topN", help="the maximum number of search hits to be"
                    " printed", type=int)
parser.add_argument("--printIdsOnly", help="print only commentIds and not ids"
                    " and their corresponding comments", action="store_true")

args = parser.parse_args()
if args.printIdsOnly:
    print("printIdsOnly turned on")
if os.path.isfile(args.query):
    print(args.query)
else:
    print(f'could not find a file called {args.query}')
