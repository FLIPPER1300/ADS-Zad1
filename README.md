# ADS – Assignment 1: Optimal Binary Search Tree

Implementation of an **Optimal Binary Search Tree (OBST)** built from word-frequency
dictionaries, developed for the *Algorithms and Data Structures* (ADS) course.

## Overview

The program reads word frequencies from two dictionary files, filters out the most
frequent words, and uses dynamic programming to construct a binary search tree that
minimizes the expected number of comparisons when searching for a key. It also
reports the search cost for a given word and visualizes the resulting tree.

## How it works

1. **Parsing** – `parse_files` reads `dictionary1.txt` and `dictionary2.txt`, where
   each line has the format `<frequency> <word>`, and merges them into a single
   frequency table.
2. **Filtering & probabilities** – Only words with a frequency greater than `40000`
   are kept as keys of the tree. Their probabilities are computed relative to the
   total frequency of *all* words in both dictionaries.
3. **Missing-key (q) probabilities** – `_compute_q_probabilities` computes the
   probability mass of all words that fall *between* two consecutive tree keys
   (as well as before the first and after the last key), representing unsuccessful
   searches.
4. **Optimal BST construction** – `_construct_optimal_bst` uses the classic
   O(n³) dynamic programming algorithm to compute the minimum expected search cost
   and the optimal root for every subinterval of keys, then builds the tree
   recursively from the root table.
5. **Search simulation** – `pocet_porovnani` (Slovak for "number of comparisons")
   searches for a given word in the constructed tree and returns the number of
   comparisons performed along with the path taken.
6. **Visualization** – `visualize_tree` draws the resulting tree using
   `networkx` and `matplotlib`.

## Project structure

| File | Description |
|---|---|
| [Zad_1.py](Zad_1.py) | Main implementation: `Node`, `OptimalBST`, file parsing, and tree visualization |
| [dictionary1.txt](dictionary1.txt) | Word frequency list (`<frequency> <word>` per line) |
| [dictionary2.txt](dictionary2.txt) | Word frequency list (`<frequency> <word>` per line) |

## Requirements

- Python 3
- `matplotlib`
- `networkx`

Install dependencies:

```bash
pip install matplotlib networkx
```

## Usage

Run the script from the project directory (it expects `dictionary1.txt` and
`dictionary2.txt` to be present in the working directory):

```bash
python Zad_1.py
```

By default the script:
- builds the optimal BST from both dictionaries,
- prints statistics (total frequency, number of tree keys, dictionary size,
  optimal cost, root key),
- searches for the word `"had"` and prints the number of comparisons and the
  search path,
- opens a plot window showing the constructed tree.

To search for a different word, change the `searched_word` variable near the
bottom of [Zad_1.py](Zad_1.py).

