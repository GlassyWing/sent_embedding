import numpy as np


def cos_sim(queries, values):
    return queries @ values.transpose()


