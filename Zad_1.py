from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx


class Node:
    def __init__(self, word, probability):
        self.word = word
        self.probability = probability
        self.left = None
        self.right = None


class OptimalBST:
    def __init__(self, words, frequencies, total_freq):
        self.words = sorted(words.items())  # Usporiadanie podľa lexikografického poradia
        self.total_freq = total_freq
        self.q_probs = self._compute_q_probabilities(frequencies)
        self.tree, self.optimal_cost = self._construct_optimal_bst()

    def _compute_q_probabilities(self, frequencies):
        q_probs = []
        bst_keys = [word for word, _ in self.words]
        all_words = sorted(frequencies.keys())  # Lexikografické zoradenie všetkých slov

        # Pre každú dvojicu kľúčov v BST hľadáme slová medzi nimi
        for i in range(len(bst_keys) - 1):
            k_i, k_next = bst_keys[i], bst_keys[i + 1]
            between_words = [w for w in all_words if k_i < w < k_next]
            between_freq = sum(frequencies[w] for w in between_words)
            q_probs.append(between_freq / self.total_freq)

        # Počiatočná a koncová pravdepodobnosť
        before_first = sum(frequencies[w] for w in all_words if w < bst_keys[0]) / self.total_freq
        after_last = sum(frequencies[w] for w in all_words if w > bst_keys[-1]) / self.total_freq

        return [before_first] + q_probs + [after_last]

    def _construct_optimal_bst(self):
        n = len(self.words)
        keys = [word for word, _ in self.words]
        probabilities = [prob for _, prob in self.words]

        # Vytvorenie matíc na uchovanie nákladov a koreňov
        cost = [[0] * (n + 1) for _ in range(n + 1)]
        root = [[0] * n for _ in range(n)]

        # Inicializácia jednobodových intervalov
        for i in range(n):
            cost[i][i] = probabilities[i] + self.q_probs[i] + self.q_probs[i + 1]
            root[i][i] = i

        # Dynamické programovanie na výpočet optimálnych nákladov
        for length in range(2, n + 1):  # Dĺžka intervalu
            for i in range(n - length + 1):
                j = i + length - 1
                cost[i][j] = float('inf')
                total_prob = sum(probabilities[i:j + 1]) + sum(self.q_probs[i:j + 2])

                for r in range(i, j + 1):
                    left_cost = cost[i][r - 1] if r > i else 0
                    right_cost = cost[r + 1][j] if r < j else 0
                    temp_cost = left_cost + right_cost + total_prob

                    if temp_cost < cost[i][j]:
                        cost[i][j] = temp_cost
                        root[i][j] = r

        return self._build_tree(keys, root, 0, n - 1), cost[0][n - 1]

    def _build_tree(self, keys, root, i, j):
        if i > j:
            return None

        optimal_root = root[i][j]
        node = Node(keys[optimal_root], self.words[optimal_root][1])
        node.left = self._build_tree(keys, root, i, optimal_root - 1)
        node.right = self._build_tree(keys, root, optimal_root + 1, j)

        return node

    def search(self, word):
        comparisons = 0
        path = []
        node = self.tree

        while node:
            comparisons += 1
            path.append(node.word)

            if word == node.word:
                return comparisons, path
            elif word < node.word:
                node = node.left
            else:
                node = node.right

        return "slovo nenájdené", comparisons, path

    def visualize_tree(self):
        def add_edges(graph, node, parent=None, pos={}, level=0, x=0, dx=1.0):
            """ Rekurzívne pridáva uzly a hrany do grafu a ukladá pozície uzlov. """
            if node:
                pos[node.word] = (x, -level)
                graph.add_node(node.word)

                if parent:
                    graph.add_edge(parent, node.word)

                # Rekurzívne pridávanie ľavého a pravého dieťaťa s posunom
                add_edges(graph, node.left, node.word, pos, level + 1, x - dx, dx / 2)
                add_edges(graph, node.right, node.word, pos, level + 1, x + dx, dx / 2)

            return pos

        graph = nx.DiGraph()
        pos = add_edges(graph, self.tree)

        plt.figure(figsize=(19, 12))

        # Vykreslenie grafu so zachovaním pozícií
        nx.draw(graph, pos, with_labels=True, node_size=2500, node_color='green',
                font_size=12, font_weight='bold', edge_color='gray')

        plt.title("Optimal Binary Search Tree", fontsize=16)
        plt.show()


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
    bst = OptimalBST(probabilities, frequencies, total_freq)

    # Výpis štatistík
    print(f"Total frequency: {total_freq}")
    print(f"Tree Keys count: {len(probabilities)}")
    print(f"Dictionary size: {len(frequencies)}")
    print(f"Optimal cost: {bst.optimal_cost}")
    print(f"Root key: {bst.tree.word if bst.tree else 'None'}")

    return bst


# Použitie:
bst = build_bst_from_files('dictionary1.txt', 'dictionary2.txt')
searched_word = "had"
result = bst.search(searched_word)
print(f"Searched word: {searched_word}")
print(f"Comparison count: {result[0]}")
print(f"Route of searching: {result[1]}")

bst.visualize_tree()
