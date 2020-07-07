import numpy as np


class WordWeight:

    def __init__(self, path):
        self.weight4ind = []
        self.word2weight = {}
        with open(path, "r", encoding="utf-8") as file:
            for ind, line in enumerate(file):
                splits = line.split()
                word = splits[0]
                weight = float(splits[1])
                self.weight4ind.append(weight)
                self.word2weight[word] = weight
        # add weight zero for mask.
        self.weight4ind.append(0)
        self.weight4ind = np.array(self.weight4ind)
