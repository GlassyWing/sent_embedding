"""
This file used to generate word-weight file.
"""

from sent_sim.utils import load_words
from tools.word_freq import WordFreq
import pkuseg
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file_path", "-vb", required=True,
                        help="Vocab file, which contains word that need to compute weight.")
    parser.add_argument("--word_freq_path", "-wf", required=True,
                        help="Word freq file, each record in that contain a word and its freq.")
    parser.add_argument("--output", "-o", required=True,
                        help="Output path.")
    args = parser.parse_args()

    seg = pkuseg.pkuseg()
    word_freq = WordFreq(seg, args.word_freq_path)
    vocab = list(load_words(args.vocab_file_path).keys())
    weights = word_freq.word_weight(vocab)

    with open(args.output, "w", encoding="utf8") as file:
        for word, weight in zip(vocab, weights):
            file.write(word + " " + str(weight) + "\n")

