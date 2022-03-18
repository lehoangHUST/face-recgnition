import numpy as np
import tensorflow as tf
from numpy.linalg import norm


# =====================================
# DISTANCES
def norm_p(emb1, emb2, p: int):
    # Type list => Numpy array
    if not isinstance(emb1, np.ndarray):
        emb1 = np.array(emb1)
    if not isinstance(emb2, np.ndarray):
        emb2 = np.array(emb2)
    return norm(emb1-emb2, p)
# =====================================
