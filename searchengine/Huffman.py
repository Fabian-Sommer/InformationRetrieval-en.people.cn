#!/usr/bin/env python3

from bitarray import bitarray as BitArray
import heapq
from Common import *


class Node():
    def init_leaf(self, symbol, weight):
        self.symbol = symbol
        self.weight = weight
        return self

    def init_parent(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child
        self.weight = self.left_child.weight + self.right_child.weight
        return self

    def is_leaf(self):
        return hasattr(self, 'symbol')

    def child(self, bit):
        assert(not self.is_leaf() and bit in (0, 1))
        if bit == 0:
            return self.left_child
        else:
            return self.right_child

    def symbols_in_subtree(self):
        if self.is_leaf():
            yield self.symbol
        else:
            for symbol in self.left_child.symbols_in_subtree():
                yield symbol
            for symbol in self.right_child.symbols_in_subtree():
                yield symbol

    def __lt__(self, other):
        return self.weight < other.weight


# returns huffman_tree_root and symbol_to_encoding_dict
def derive_encoding(symbol_to_frequency_dict):
    assert(len(symbol_to_frequency_dict) > 0)
    symbol_to_encoding_dict = \
        dict(((symbol, '') for symbol in symbol_to_frequency_dict.keys()))
    heap = [Node().init_leaf(symbol, frequency)
            for symbol, frequency in symbol_to_frequency_dict.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left_child = heapq.heappop(heap)
        right_child = heapq.heappop(heap)
        heapq.heappush(heap, Node().init_parent(left_child, right_child))

        for symbol in left_child.symbols_in_subtree():
            symbol_to_encoding_dict[symbol] = \
                '0' + symbol_to_encoding_dict[symbol]
        for symbol in right_child.symbols_in_subtree():
            symbol_to_encoding_dict[symbol] = \
                '1' + symbol_to_encoding_dict[symbol]

    for key, value in symbol_to_encoding_dict.items():
        symbol_to_encoding_dict[key] = BitArray(value)
    return symbol_to_encoding_dict


# returns bytes, leading 1s and first 0 are padding:
# 11110011 -> 11110 = padding, 011 = data
def encode(string, symbol_to_encoding_dict):
    string_len = 0
    for symbol in string:
        string_len += len(symbol_to_encoding_dict[symbol])
    padding = 8 - (string_len % 8)
    bit_array = BitArray((padding - 1) * '1' + '0')
    for symbol in string:
        bit_array += BitArray(symbol_to_encoding_dict[symbol])
    return bit_array.tobytes()


def decode(binary_data, symbol_to_encoding_dict):
    padding = 0
    bit_array = BitArray()
    bit_array.frombytes(binary_data)
    while bool(bit_array[padding]):
        padding += 1
    decoded_chars = bit_array[padding+1:].decode(symbol_to_encoding_dict)
    return ''.join(decoded_chars)


if __name__ == '__main__':
    string = 'hallo'
    symbol_to_frequency_dict = {'h': 120, 'a': 20, 'l': 2, 'o': 73}
    symbol_to_encoding_dict = derive_encoding(symbol_to_frequency_dict)
    encoded_string = encode('hallo', symbol_to_encoding_dict)
    decoded_string = decode(encoded_string, symbol_to_encoding_dict)

    print(f'string: {string}')
    print(f'symbol_to_encoding_dict: {symbol_to_encoding_dict}')
    print(f'encoded: {encoded_string}')
    print(f'decoded: {decoded_string}')
