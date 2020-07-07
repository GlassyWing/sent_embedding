import numpy as np

def cos_sim(queries, values):
    queries = queries / np.linalg.norm(queries, axis=-1, keepdims=True)
    values = values / np.linalg.norm(values, axis=-1, keepdims=True)
    return queries @ values.transpose()


