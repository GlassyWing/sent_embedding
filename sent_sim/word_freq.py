from collections import Counter
import math


def zipf_to_freq(zipf):
    """
    Convert a word frequency from the Zipf scale to a proportion between 0 and
    1.
    The Zipf scale is a logarithmic frequency scale proposed by Marc Brysbaert,
    who compiled the SUBTLEX data. The goal of the Zipf scale is to map
    reasonable word frequencies to understandable, small positive numbers.
    A word rates as x on the Zipf scale when it occurs 10**x times per billion
    words. For example, a word that occurs once per million words is at 3.0 on
    the Zipf scale.
    """
    return 10 ** zipf / 1e9


def freq_to_zipf(freq):
    """
    Convert a word frequency from a proportion between 0 and 1 to the
    Zipf scale (see `zipf_to_freq`).
    """
    return math.log(freq, 10) + 9


class WordFreq:

    def __init__(self, seg, dict_path, weightpara=1e-4):
        self.seg = seg
        self.counter = Counter()
        self.weightpara = weightpara
        self.total_freqs = 0
        with open(dict_path, "r", encoding="utf-8") as file:
            for line in file:
                splits = line.split()
                word = splits[0]
                freq = float(splits[1])
                self.counter[word] = freq

    def word_weight(self, words):
        if isinstance(words, list):
            word_weights = []
            for word in words:
                word_freq = self.word_frequency(word)
                word_weight = self.weightpara / (self.weightpara + word_freq)
                word_weights.append(word_weight)
            return word_weights
        else:
            word_freq = self.word_frequency(words)
            word_weight = self.weightpara / \
                          (self.weightpara + word_freq)
            return word_weight

    def word_frequency(self, word, minimum=0.):
        tokens = self.seg.cut(word)
        if tokens is None or len(tokens) == 0:
            return minimum

        one_over_result = 0.0
        for token in tokens:
            if token not in self.counter:
                return minimum
            one_over_result += 1.0 / \
                               (self.counter.get(token))

        freq = 1.0 / one_over_result
        unrounded = max(freq, minimum)

        if unrounded != 0.:
            leading_zeros = math.floor(-math.log(unrounded, 10))
            rounded = round(unrounded, leading_zeros + 3)
        else:
            rounded = 0.

        return rounded

    def zipf_frequency(self, word, minimum=0.):
        freq_min = zipf_to_freq(minimum)
        freq = self.word_frequency(word, minimum=freq_min)
        return round(freq_to_zipf(freq), 2)

if __name__ == '__main__':
    import pkuseg
    import time
    seg = pkuseg.pkuseg()
    start_time = time.time()
    word_freq = WordFreq(seg, dict_path="../dict/word_freq.txt")
    print(time.time() - start_time)
    print(word_freq.zipf_frequency("北京地铁"))
    print(word_freq.zipf_frequency("地铁"))
    print(word_freq.zipf_frequency("北京"))