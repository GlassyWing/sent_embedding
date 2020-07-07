def load_sentences(file_path):
    sentences = []
    with open(file_path, 'r', encoding="utf-8") as file:
        for no, line in enumerate(file):
            if len(line.strip()) == 0:
                continue
            sentences.append(line.rstrip('\n'))
    return sentences


def load_words(word_dict_path):
    words = {}
    with open(word_dict_path, 'r', encoding="utf-8") as file:
        no = 0
        for line in file:
            words[line.rstrip("\n")] = no
            no += 1
    return words
