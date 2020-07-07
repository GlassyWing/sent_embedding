import logging

import h5py as h5
import numpy as np
import argparse
import os


def getWordmap(textfile):
    words = {}
    We = []

    with open(textfile, 'r', encoding='utf-8') as f:
        n = 0
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)
            if coefs.shape[0] == 300:
                words[word] = n
                We.append(coefs)
                n += 1

    return words, np.stack(We, axis=0)


def create_h5(file_path, We):
    f = h5.File(file_path, "w")
    f.create_dataset("emb", shape=We.shape)
    f["emb"][:] = We
    f.close()


def create_dict(file_path, words):
    with open(file_path, "w", encoding="utf-8") as file:
        for word in words:
            file.write(word + "\n")


if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument("--word_emb_path", "-i", required=True, help="Word embedding file path.")
    parser.add_argument("--output_dir", "-o", required=True, help="Output Directory.")

    args = parser.parse_args()
    words, We = getWordmap(args.word_emb_path)

    output_dir = args.output_dir
    emb_path = os.path.join(output_dir, "word.embedding.h5")
    dict_path = os.path.join(output_dir, "vocab.txt")

    logging.info("Creating word dictionary.")
    create_dict(dict_path, words)
    logging.info("Creating word embedding.")
    create_h5(emb_path, We)
    logging.info("Done.")
