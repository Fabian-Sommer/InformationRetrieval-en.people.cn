import bitstring
import heapq

class HuffmanNode():
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
        elif bit == 1:
            return self.right_child

    def symbols_in_subtree(self):
        if self.is_leaf():
            return [ self.symbol ]
        else:
            return self.left_child.symbols_in_subtree() + self.right_child.symbols_in_subtree()

    def __lt__(self, other):
        return self.weight < other.weight

# returns huffman_tree_root and symbol_to_encoding_dict
def derive_huffman_encoding(symbol_to_frequency_dict):
    symbol_to_encoding_dict = dict(( (symbol, '') for symbol in symbol_to_frequency_dict.keys() ))
    heap = [
        HuffmanNode().init_leaf(symbol, frequency)
        for symbol, frequency in symbol_to_frequency_dict.items()
    ]
    heapq.heapify(heap)
    while len(heap) > 1:
        left_child = heapq.heappop(heap)
        right_child = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode().init_parent(left_child, right_child))

        for symbol in left_child.symbols_in_subtree():
            symbol_to_encoding_dict[symbol] = '0' + symbol_to_encoding_dict[symbol]
        for symbol in right_child.symbols_in_subtree():
            symbol_to_encoding_dict[symbol] = '1' + symbol_to_encoding_dict[symbol]

    return heap[0], symbol_to_encoding_dict

def encode_huffman(string, symbol_to_encoding_dict):
    encoded_string = '0b'
    for symbol in string:
        encoded_string += symbol_to_encoding_dict[symbol]
    return encoded_string

def decode_huffman(bit_string, huffman_tree_root):
    decoded_string = ''
    bit_stream = bitstring.ConstBitStream(bin=bit_string)
    while bit_stream.pos < len(bit_stream):
        node = huffman_tree_root
        while not node.is_leaf():
            node = node.child(bit_stream.read('bool'))
        decoded_string += node.symbol
    return decoded_string

if __name__ == '__main__':
    string = 'hallo'
    symbol_to_frequency_dict = { 'h': 120, 'a': 20, 'l': 2, 'o': 73 }
    huffman_tree_root, symbol_to_encoding_dict = derive_huffman_encoding(symbol_to_frequency_dict)
    encoded_string = encode_huffman('hallo', symbol_to_encoding_dict)
    decoded_string = decode_huffman(encoded_string, huffman_tree_root)

    print(f'string: {string}')
    print(f'symbol_to_encoding_dict: {symbol_to_encoding_dict}')
    print(f'encoded: {encoded_string}')
    print(f'decoded: {decoded_string}')