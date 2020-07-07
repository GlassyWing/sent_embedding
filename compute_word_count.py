"""
This file used to generate word-freq file.
"""

import argparse
import logging
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import reduce


def count_file(file_path):
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    counter = Counter()

    lines = 0
    with open(file_path, 'r', encoding="utf-8") as file:

        for line in file:
            if len(line.strip()) == 0:
                continue

            tokens = line.split()

            new_line = []
            for token in tokens:

                if re.match(r'\W+', token):
                    continue
                new_line.append(token)
            counter.update(new_line)
            lines += 1
            logging.info("pid:" + str(os.getpid()) + ":" + str(lines))

    return counter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", help="Origin word-freq file path.")
    parser.add_argument("--path", "-i", required=True, help="Input path.")
    parser.add_argument("--output", "-o", required=True, help="Output path.")
    args = parser.parse_args()

    paths = args.path.split(",")

    if len(paths) > 1:
        with ProcessPoolExecutor() as executor:
            counters = []
            for counter in executor.map(count_file, paths):
                counters.append(counter)

            counter = reduce(lambda a,b: a.update(b), counters)
    else:
        counter = count_file(paths[0])

    if args.origin is not None:
        origin_counter = Counter()
        with open(args.origin, "r", encoding="utf-8") as file:
            for line in file:
                splits = line.split()
                word = splits[0]
                freq = float(splits[1])
                origin_counter[word] = freq

        counter.update(origin_counter)

    with open(args.output, "w", encoding="utf-8") as file:
        for k, v in sorted(counter.items(), key=lambda item: item[1]):
            file.write(k + "\t" + str(v) + "\n")
