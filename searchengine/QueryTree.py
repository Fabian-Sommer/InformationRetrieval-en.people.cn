#!/usr/bin/env python3

import re


class TokenNode():
    def __init__(self, raw_query_token):
        assert(len(raw_query_token) > 0)
        self.raw_query_token = raw_query_token
        parts = raw_query_token.strip().rpartition('NOT ')
        self.is_negated = parts[1] == 'NOT '
        query_token = parts[2].lower()
        # self.result = []
        if len(query_token) > 1 and query_token[-2] == "'":
            assert(query_token[0] == "'" and query_token.count("'") == 2
                   and query_token[-1] == '*')
            self.kind = 'phrase_prefix'
            parts = query_token[1:-2].rpartition(' ')
            self.phrase_start = parts[0]
            self.prefix = parts[2]
            self.query_token = query_token[1:-2]
        elif query_token[-1] == "'":
            assert(query_token[0] == "'" and query_token.count("'") == 2)
            self.kind = 'phrase'
            self.phrase = query_token[1:-1]
            self.query_token = self.phrase
        elif query_token[-1] == '*':
            assert(query_token.count('*') == 1)
            self.kind = 'prefix'
            self.prefix = query_token[:-1]
            self.query_token = self.prefix
        elif 'replyto:' in query_token:
            assert(query_token.count('replyto:') == 1 and
                   query_token.startswith('replyto:'))
            self.kind = 'reply_to'
            self.target_cid = int(query_token.partition('replyto:')[2])
            self.query_token = ''
        else:
            assert(' ' not in query_token)
            self.kind = 'keyword'
            self.keyword = query_token
            self.query_token = self.keyword
        pass
    # def resolve(self, search_function):
    #     pass

    def __repr__(self):
        return f'TokenNode("{self.raw_query_token}")'


class AndNode():
    def __init__(self, children):
        assert(len(children) >= 1)

        # if all children are negated
        if not next(filter(lambda child: not child.is_negated, children),
                    False):
            raise RuntimeError('at least one token must not be negated '
                               '(NOT) in an AND')

        self.children = children
        self.is_negated = False
        # self.result = []

    # def resolve(self):
    #     result = set()
    #     to_be_removed = []
    #     for child in self.children:
    #         child_result, negated = child.resolve()
    #         if negated:
    #             to_be_removed.append(child_result)
    #         else:
    #             result.update(child_result)
    #     result.difference_update(*to_be_removed)
    #     return result, self.is_negated

    def __repr__(self):
        return f'AndNode({self.children})'


class OrNode():
    def __init__(self, children):
        assert(len(children) >= 1)

        if next(filter(lambda child: child.is_negated, children), False):
            raise RuntimeError('negated tokens must not be used with AND')

        self.children = children
        self.is_negated = False
        # self.result = []

    # def resolve(self):
    #     result = set()
    #     for child in self.children:
    #         child_result, negated = child.resolve()
    #         if negated:
    #             print(f'WARNING: ignoring {child.query_token}'
    #                   'NOT always has to be used with AND')
    #         else:
    #             result.update(child_result)
    #     return result, self.is_negated

    def __repr__(self):
        return f'OrNode({self.children})'


# query tokens delimited by space, for non boolean queries
class SpaceNode():
    def __init__(self, children):
        assert(len(children) >= 1)
        self.children = children
        self.is_negated = False
        self.query_terms = []
        for child in children:
            if child.query_token != '':
                self.query_terms.extend(child.query_token.split(' '))

    def __repr__(self):
        return f'SpaceNode({self.children})'


def build(query):
    query = query.strip()
    root_node = None
    if ' AND ' in query or ' OR ' in query or 'NOT ' in query:  # boolean query
        query_list = []
        query = re.sub('(?<! AND) NOT ', ' AND NOT ', query)
        assert(query.count("'") % 2 == 0)

        # initialize, create TokenNodes
        for match in re.findall("([^']+)|('[^']+'\*?)", query):
            print(match[0], match[1])
            if match[1] != '':  # second capture group ('[^']+'\*?)
                query_list.append(TokenNode(match[1]))
            else:  # first capture group ([^']+)
                for query_token in re.findall('(?:NOT )?[^ ]+', match[0]):
                    if query_token in ('AND', 'OR'):
                        query_list.append(query_token)
                    else:
                        query_list.append(TokenNode(query_token))

        # create AndNodes
        query_list_AND = []

        positions_OR = (i for i in range(len(query_list) + 1)
                        if i == len(query_list) or query_list[i] == 'OR')
        last_position_OR = -1
        for position_OR in positions_OR:
            list_AND = query_list[last_position_OR + 1:position_OR]
            last_position_OR = position_OR
            nodes_connected_by_AND = list_AND[::2]
            query_list_AND.append(AndNode(nodes_connected_by_AND))
        root_node = OrNode(query_list_AND)
        root_node.is_boolean_query = True

    else:  # non boolean query
        query_list = []
        for match in re.findall("([^']+)|('[^']+'\*?)", query):
            if match[1] != '':  # second capture group ('[^']+'\*?)
                query_list.append(TokenNode(match[1]))
            else:  # first capture group ([^']+)
                for query_token in re.findall('[^ ]+', match[0]):
                    query_list.append(TokenNode(query_token))
        root_node = SpaceNode(query_list)
        root_node.is_boolean_query = False
    return root_node


if __name__ == '__main__':
    print(build("NOT merkel NOT xi OR 'something something'"))
