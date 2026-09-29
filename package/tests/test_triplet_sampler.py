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


class TestSaveTriplets(unittest.TestCase):

    def test_save_triplets_leaves_default_split_names_untouched(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            profiles = pd.DataFrame(np.eye(4, dtype=int),
                                    index=['A', 'B', 'C', 'D'])
            sampler = AdriaTripletSampler(
                FakeSignature(profiles), FakeSignature(profiles),
                os.path.join(tmp_dir, 'traintest.h5'))
            # the old code reversed the default list in place
            sampler.save_triplets(np.array([[0, 1, 2]] * 10))
            defaults = AdriaTripletSampler.save_triplets.__defaults__
            self.assertIn(['train', 'test'], defaults)
        finally:
            shutil.rmtree(tmp_dir)
