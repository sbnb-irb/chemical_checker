import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

from chemicalchecker.util.splitter import AdriaTripletSampler


class FakeSignature(object):
    """Minimal stand-in for the signature methods the sampler uses."""

    def __init__(self, df, data_path='fake.h5'):
        self.df = df
        self.keys = df.index.values
        self.data_path = data_path

    def as_dataframe(self):
        return self.df

    def get_vectors(self, keys, dataset_name='x'):
        return keys, self.df.loc[keys].values.astype(float)


def binary_profiles(n_mols=60, n_features=20, n_inactive=3, seed=0):
    """Binary profiles in 3 clusters, plus some molecules without effect."""
    rng = np.random.default_rng(seed)
    centers = rng.random((3, n_features)) < 0.3
    rows = []
    for i in range(n_mols - n_inactive):
        flips = rng.random(n_features) < 0.1
        rows.append(np.logical_xor(centers[i % 3], flips))
    rows += [np.zeros(n_features, dtype=bool)] * n_inactive
    keys = ['MOL%03d' % i for i in range(n_mols)]
    return pd.DataFrame(np.array(rows, dtype=int), index=keys)


class TestAdriaTripletSampler(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.out_file = os.path.join(self.tmp_dir, 'traintest.h5')
        self.profiles = binary_profiles()
        self.mol_signature = FakeSignature(
            pd.DataFrame(np.random.default_rng(1).random((60, 8)),
                         index=self.profiles.index))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def sample(self, profiles, **kwargs):
        sampler = AdriaTripletSampler(FakeSignature(profiles),
                                      self.mol_signature, self.out_file)
        kwargs.setdefault('num_triplets', 3000)
        kwargs.setdefault('plot', False)
        sampler.generate_triplets(**kwargs)
        with h5py.File(self.out_file, 'r') as fh:
            return np.vstack([fh[name][:] for name in fh
                              if name.startswith('t_')])

    def test_triplets_follow_the_binary_similarity(self):
        triplets = self.sample(self.profiles, seed=0)
        self.assertGreater(len(triplets), 2500)
        self.assertTrue(np.all(triplets[:, 0] != triplets[:, 1]))
        self.assertTrue(np.all(triplets[:, 0] != triplets[:, 2]))
        # anchors and positives come from the same cluster far more often
        # than anchors and negatives
        cluster = np.arange(60) % 3
        same_pos = np.mean(cluster[triplets[:, 0]] == cluster[triplets[:, 1]])
        same_neg = np.mean(cluster[triplets[:, 0]] == cluster[triplets[:, 2]])
        self.assertGreater(same_pos, 0.9)
        self.assertLess(same_neg, 0.5)

    def test_molecules_without_effect_get_no_triplets(self):
        triplets = self.sample(self.profiles, seed=0)
        inactive = np.arange(57, 60)
        self.assertFalse(np.isin(triplets, inactive).any())

    def test_seed_makes_sampling_reproducible(self):
        np.random.seed(0)  # save_triplets splits with the global generator
        first = self.sample(self.profiles, seed=3)
        np.random.seed(0)
        second = self.sample(self.profiles, seed=3)
        np.testing.assert_array_equal(first, second)

    def test_continuous_signature_is_rejected(self):
        continuous = self.profiles + 0.5
        with self.assertRaisesRegex(ValueError, 'needs a binary'):
            self.sample(continuous)

    def test_missing_molecules_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing from the triplet'):
            self.sample(self.profiles.iloc[5:])

    def test_plot_is_saved(self):
        self.sample(self.profiles, seed=0, plot=True)
        self.assertTrue(os.path.isfile(self.out_file + '.png'))

    def test_save_triplets_leaves_default_split_names_untouched(self):
        sampler = AdriaTripletSampler(FakeSignature(self.profiles),
                                      self.mol_signature, self.out_file)
        triplets = np.array([[0, 1, 2]] * 10)
        sampler.save_triplets(triplets)  # the old code reversed this in place
        defaults = AdriaTripletSampler.save_triplets.__defaults__
        self.assertIn(['train', 'test'], defaults)
