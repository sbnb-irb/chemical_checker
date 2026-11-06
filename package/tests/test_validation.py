import os
import shutil
import tempfile
import unittest
import numpy as np
from scipy.spatial.distance import cosine

from chemicalchecker.util.plot import Plot


class MockSign(object):
    """Minimal signature exposing what `Plot.vector_validation` reads."""

    def __init__(self, keys, V):
        self.keys = np.array(keys)
        self.V = V
        self.unique_keys = set(keys)

    def get_vectors(self, keys):
        idx = {k: i for i, k in enumerate(self.keys)}
        keys = sorted(keys)
        return np.array(keys), self.V[[idx[k] for k in keys]]


class TestValidation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # two groups of molecules; dense int8 counts like C5.001 sign0,
        # large enough that int8 dot products overflow
        rng = np.random.RandomState(42)
        n_feat, n_per_group = 500, 10
        V, keys = [], []
        for group in range(2):
            profile = np.zeros(n_feat)
            profile[group * 250:(group + 1) * 250] = 15
            for i in range(n_per_group):
                noise = rng.randint(0, 6, n_feat)
                V.append(np.clip(profile + noise, 0, 20))
                keys.append("MOL%d%02d" % (group, i))
        self.V_int8 = np.array(V, dtype=np.int8)
        self.keys = keys
        with open(os.path.join(self.tmp, "moa_validation.tsv"), "w") as fh:
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    same = int(keys[i][3] == keys[j][3])
                    fh.write("%s\t%s\t%d\n" % (keys[i], keys[j], same))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_int8_overflow_precondition(self):
        # scipy on int8 must be corrupted for this data, or the test below
        # would pass even without the fix
        u, v = self.V_int8[0], self.V_int8[1]
        self.assertNotAlmostEqual(
            cosine(u, v), cosine(u.astype(float), v.astype(float)), places=3)

    def test_vector_validation_integer_dtype(self):
        plot = Plot("C5.001", self.tmp, self.tmp)
        _, auc_int8, cov_int8 = plot.vector_validation(
            MockSign(self.keys, self.V_int8), "sign0", prefix="moa")
        _, auc_float, cov_float = plot.vector_validation(
            MockSign(self.keys, self.V_int8.astype(np.float64)), "sign0",
            prefix="moa")
        self.assertAlmostEqual(auc_int8, auc_float, places=10)
        self.assertEqual(cov_int8, cov_float)
        # groups are well separated -> near-perfect AUC
        self.assertGreater(auc_int8, 0.99)
        # non-negative data: every cosine distance lies in [0, 1]
        S = np.loadtxt(os.path.join(
            self.tmp, "moa_sign0_ks_validation_S.tsv"))[:, 0]
        self.assertTrue(np.all((S >= 0) & (S <= 1 + 1e-9)))
