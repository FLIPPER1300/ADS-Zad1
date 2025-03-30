import heapq
from collections import defaultdict


class OptimalBST:
    def __init__(self, words):
        self.words = words
        self.tree = self._construct_optimal_bst()

    def _construct_optimal_bst(self):
        # Použitie algoritmu na zostavenie optimálneho BST
        # Implementácia dynamického programovania z CLRS
        pass

    def search(self, word):
        comparisons = 0
        path = []
        node = self.tree

        while node:
            comparisons += 1
            path.append(node["word"])

            if word == node["word"]:
                return comparisons, path
            elif word < node["word"]:
                node = node["left"]
            else:
                node = node["right"]

        return "slovo nenájdené", comparisons, path


def parse_files(file1, file2):
    frequencies = defaultdict(int)

    for file in [file1, file2]:
        with open(file, 'r') as f:
            for line in f:
                freq, word = line.strip().split()
                frequencies[word] += int(freq)

    return frequencies


def build_bst_from_files(file1, file2):
    frequencies = parse_files(file1, file2)

    # Filtrovanie slov s frekvenciou > 40 000
    filtered_words = {word: freq for word, freq in frequencies.items() if freq > 40000}

    # Výpočet pravdepodobností
    total_freq = sum(frequencies.values())
    probabilities = {word: freq / total_freq for word, freq in filtered_words.items()}

    # Zostrojenie BST
    bst = OptimalBST(probabilities)
    return bst


# Použitie:
bst = build_bst_from_files('dictionary1.txt', 'dictionary2.txt')
result = bst.search("example")
print(result)
