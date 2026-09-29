"""Splitter on Neighbor triplets."""
import os
import h5py
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.preprocessing import RobustScaler

from chemicalchecker.util import logged
from chemicalchecker.util.remove_near_duplicates import RNDuplicates


@logged
class TripletIterator(object):
    """TripletIterator class."""

    def __init__(self, hdf5_file, split, replace_nan=None):
        """Initialize a TripletIterator instance.

        This allows iterating on train/test splits of triplets 
        generated via a `TripletSampler` class.
        We assume the file is containing different splits.
        e.g. "x_train", "y_train", "x_test", ...

        Args:
            hdf5_file(str): the path to a file generated via 
                TripletSample class.
            split(str): The H5 typically contains 'train' or
                'test' splits, the iterator will focus on
                that split.
            replace_nan(float): If None, nothing is replaced,
                otherwise NaN are replaced by the value
                specified. 
        """
        self._file = hdf5_file
        self._f = None
        self.replace_nan = replace_nan
        self.x_name = "x"
        if split is None:
            self.t_name = "t"
            self.y_name = "y"
        else:
            self.t_name = "t_%s" % split
            self.y_name = "y_%s" % split
            available_splits = self.get_split_names()
            if split not in available_splits:
                raise Exception("Split '%s' not found in %s!" %
                                (split, str(available_splits)))

    def get_ty_shapes(self):
        """Return the shapes of X an Y."""
        self.open()
        t_shape = self._f[self.t_name].shape
        y_shape = self._f[self.y_name].shape
        self.close()
        return t_shape, y_shape

    def get_xy_shapes(self):
        """Return the shapes of X an Y."""
        self.open()
        x_shape = self._f[self.x_name].shape
        self.close()
        return x_shape

    def get_split_names(self):
        """Return the name of the splits."""
        self.open()
        if "split_names" in self._f:
            split_names = [a.decode() for a in self._f["split_names"]]
        else:
            split_names = ['train', 'test']
        self.close()
        combos = itertools.combinations_with_replacement(split_names, 2)
        return ['_'.join(x) for x in combos]

    def open(self):
        """Open the HDF5."""
        self._f = h5py.File(self._file, 'r')
        # self.__log.info("HDF5 open %s", self._file)

    def close(self):
        """Close the HDF5."""
        try:
            self._f.close()
            # self.__log.info("HDF5 close %s", self._file)
        except AttributeError:
            self.__log.error('HDF5 file is not open yet.')

    def get_t(self, beg_idx, end_idx):
        """Get a batch of X."""
        features = self._f[self.t_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_y(self, beg_idx, end_idx):
        """Get a batch of Y."""
        features = self._f[self.y_name][beg_idx: end_idx]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_x(self):
        """Get full X."""
        features = self._f[self.x_name][:]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_x_columns(self, mask):
        """Get full X."""
        features = self._f[self.x_name][:, mask]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_p(self):
        """Get full X."""
        features = self._f[self.t_name][:]
        # handle NaNs
        if self.replace_nan is not None:
            features[np.where(np.isnan(features))] = self.replace_nan
        return features

    def get_all_y(self):
        """Get full Y."""
        labels = self._f[self.y_name][:]
        return labels

    @staticmethod
    def generator_fn(file_name, split, replace_nan=None,
                     batch_size=None, shuffle=True,
                     train=True, augment_fn=None, augment_kwargs={},
                     mask_fn=None,  trim_mask=None,
                     sharedx=None, sharedx_trim=None,
                     onlyself_notself=False, p_self_decay=False):
        """Return the generator function that iterates on batches.

        A TripletIterator on the specified file and split is initialized,
        we allow for additional masking, shared X matrix and more.

        Args:
            file_name(str): The H5 generated via a `TripletSampler` class
            split(str): One of 'train_train', 'train_test', or 'test_test'
            replace_nan(bool): Value used for NaN replacement.
            batch_size(int): Size of a batch of data.
            shuffle(bool): Shuffle the order of batches.
            train(bool): At train time the augment function is applied.
            augment_fn(func): Function to augment data.
            augment_kwargs(dict): Parameters for the augment function.
            mask_fn(func): Function to mask data while iterating.
            trim_mask(array): Initial masking of data (spaces are excluded).
            sharedx(matrix): The preloaded X matrix.
            sharedx_trim(matrix): The preloaded and pre-trimmed X matrix.
            onlyself_notself(bool): when True the iterator will return a 
                quintuplet with also only_self and not_self.
            p_self_decay(bool): when True the p_self probability will decay 
                within the batch, and restart at each batch.
        """
        def notself(idxs, x1_data):
            x1_data_transf = np.copy(x1_data)
            for idx in idxs:
                # set current space to nan
                col_slice = slice(idx * 128, (idx + 1) * 128)
                x1_data_transf[:, col_slice] = np.nan
            return x1_data_transf

        TripletIterator.__log.debug('Generator for %s' % split)
        reader = TripletIterator(file_name, split)
        reader.open()
        # read shapes
        t_shape = reader._f[reader.t_name].shape
        # read data types
        x_dtype = reader._f[reader.x_name].dtype
        # no batch size -> return everything
        if not batch_size:
            batch_size = t_shape[0]
        # keep X in memory for resolving triplets quickly
        if sharedx is not None:
            if trim_mask is None:
                X = sharedx
            else:
                if sharedx_trim is not None:
                    X = sharedx_trim
                else:
                    X = sharedx[:, np.argwhere(
                        np.repeat(trim_mask, 128)).ravel()]
        else:
            TripletIterator.__log.debug('Reading X in memory')
            if trim_mask is None:
                X = reader.get_all_x()
            else:
                if sharedx_trim is not None:
                    X = sharedx_trim
                else:
                    X = reader.get_x_columns(np.argwhere(
                        np.repeat(trim_mask, 128)).ravel())
        TripletIterator.__log.debug('X shape: %s' % str(X.shape))
        # default mask is not masking
        if mask_fn is None:
            def mask_fn(*data):
                return data
        # default augment is doing nothing
        if augment_fn is None:
            def augment_fn(*data, **kwargs):
                return data
        # this variable is going to be used to shuffle batches
        batch_beg_end = np.zeros((int(np.ceil(t_shape[0] / batch_size)), 2))
        last = 0
        for row in batch_beg_end:
            row[0] = last
            row[1] = last + batch_size
            last = row[1]
        batch_beg_end = batch_beg_end.astype(int)
        # handle arguments for additional Xs
        if onlyself_notself:
            only_args = augment_kwargs.copy()
            only_args['p_only_self'] = 1.0

        TripletIterator.__log.debug(
            'Generator ready, onlyself_notself %s' % onlyself_notself)

        def example_generator_fn():
            """Generator function yields data in batches"""
            batch_kwargs = augment_kwargs.copy()
            if p_self_decay:
                # we leave a p_self > 0 for the first 10th of batches
                # then it will linearly decrease
                nr_steps = int(len(batch_beg_end) / 10) + 1
                p_self_current = augment_kwargs.get("p_self", 0.1)
                decay_step = p_self_current / nr_steps

            epoch = 0
            batch_idx = 0
            while True:
                # here we handles what happens at the last batch
                if batch_idx == len(batch_beg_end):
                    batch_idx = 0
                    epoch += 1
                    if shuffle:
                        np.random.shuffle(batch_beg_end)
                    if p_self_decay:
                        p_self_current = augment_kwargs.get("p_self", 0.1)
                # select the batch start/end and fetch triplets
                beg_idx, end_idx = batch_beg_end[batch_idx]
                tripets = reader.get_t(beg_idx, end_idx)
                y = reader.get_y(beg_idx, end_idx)
                x1 = X[tripets[:, 0]]
                x2 = X[tripets[:, 1]]
                x3 = X[tripets[:, 2]]
                if train and onlyself_notself:
                    # at train time we want to apply subsampling
                    x1 = augment_fn(x1, **batch_kwargs)
                    x2 = augment_fn(x2, **batch_kwargs)
                    x3 = augment_fn(x3, **batch_kwargs)
                if onlyself_notself:
                    x4 = augment_fn(X[tripets[:, 0]], **only_args)
                    x5 = notself(augment_kwargs['dataset_idx'], x1)
                # apply the mask function
                x1, x2, x3 = mask_fn(x1, x2, x3)
                # replace NaNs with specified value
                if replace_nan is not None:
                    x1[np.where(np.isnan(x1))] = replace_nan
                    x2[np.where(np.isnan(x2))] = replace_nan
                    x3[np.where(np.isnan(x3))] = replace_nan
                    if onlyself_notself:
                        x4[np.where(np.isnan(x4))] = replace_nan
                        x5[np.where(np.isnan(x5))] = replace_nan
                # yield the triplets
                if onlyself_notself:
                    yield [x1, x2, x3, x4, x5], y
                else:
                    yield [x1, x2, x3], y
                # go to next batch
                batch_idx += 1
                # update subsampling parameters
                if p_self_decay:
                    p_self_current -= decay_step
                    batch_kwargs['p_self'] = max(0.0, p_self_current)

        # return shapes and dtypes along with iterator
        triplet_shape = (t_shape[0], X.shape[1])
        shapes = (triplet_shape, triplet_shape, triplet_shape, triplet_shape)
        dtypes = (x_dtype, x_dtype, x_dtype, x_dtype)
        return shapes, dtypes, example_generator_fn


@logged
class BaseTripletSampler(object):
    """Base class for triplet samplers."""

    def __init__(self, triplet_signature, mol_signature, out_file,
                 save_kwargs={}):
        self.triplet_signature = triplet_signature
        self.mol_signature = mol_signature
        self.out_file = out_file
        def_save_kwargs = {
            'mean_center_x': True,
            'shuffle': True,
            'split_names': ['train', 'test'],
            'split_fractions': [.8, .2],
            'suffix': 'eval',
            'cpu': 1,
            'x_dtype': float,
            'y_dtype': float
        }
        def_save_kwargs.update(save_kwargs)
        self.save_kwargs = def_save_kwargs

    def get_split_indeces(self, rows, fractions):
        """Get random indexes for different splits."""
        if not sum(fractions) == 1.0:
            raise Exception("Split fractions should sum to 1.0")
        # shuffle indexes
        idxs = list(range(rows))
        np.random.shuffle(idxs)
        # from frequencies to indices
        splits = np.cumsum(fractions)
        splits = splits[:-1]
        splits *= len(idxs)
        splits = splits.round().astype(int)
        return np.split(idxs, splits)

    def save_triplets(self, triplets, mean_center_x=True, shuffle=True,
                      split_names=['train', 'test'],
                      split_fractions=[.8, .2],
                      suffix='eval', cpu=1,
                      x_dtype=float, y_dtype=float):
        """Save sampled triplets to file.

        This function saves triplets performing the train test split,
        shuffling and normalization.

        Args:
            triplets(array): Indexes of anchor, positive and negative for
                each triplet.
            mean_center_x(bool): Normalize data columns wise.
            shuffle(bool): shuffle order of triplets.
            split_names(list str): names of the splits.
            split_fractions(list float): fraction of each split.
            suffix(str): suffix of the generated scaler.
        """
        ink_keys = self.mol_signature.keys
        _, X = self.mol_signature.get_vectors(ink_keys, dataset_name='x')
        self.__log.debug('X.shape %s', str(X.shape))
        self.__log.debug('triplets.shape %s', str(triplets.shape))

        # mean centering features
        if mean_center_x:
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            if suffix is None:
                scaler_file = os.path.join(os.path.split(self.out_file)[0],
                                           'scaler.pkl')
            else:
                scaler_file = os.path.join(os.path.split(self.out_file)[0],
                                           'scaler_%s.pkl' % suffix)
            pickle.dump(scaler, open(scaler_file, 'wb'))

        # shuffling
        shuffle_idxs = np.arange(triplets.shape[0])
        if shuffle:
            np.random.shuffle(shuffle_idxs)
        triplets = np.array(triplets)[shuffle_idxs]

        # do train-test split on keys
        split_idxs = self.get_split_indeces(
            X.shape[0], split_fractions)

        # do train-test split for triplets (np.unique of indexes)
        split_idxs = dict(zip(split_names, split_idxs))

        # find triplets having test-test train-train and train-test
        combos = itertools.combinations_with_replacement(split_names, 2)

        # reverse split names to first write test keys (on a copy, so the
        # caller's list and the default argument are left untouched)
        split_names = list(reversed(split_names))

        # create output file
        self.__log.info('Saving Triplets to %s', self.out_file)
        with h5py.File(self.out_file, "w") as fh:
            for split_n in split_names:
                fh.create_dataset('keys_%s' % split_n,
                                  data=np.array(ink_keys[split_idxs[split_n]],
                                                dtype=h5py.string_dtype()))
            if mean_center_x:
                fh.create_dataset('scaler',
                                  data=np.array([scaler_file],
                                                dtype=h5py.string_dtype()))

            fh.create_dataset('x', data=X, dtype=x_dtype)
            fh.create_dataset('x_ink',
                              data=np.array(ink_keys,
                                            dtype=h5py.string_dtype()))

            for split1, split2 in combos:
                split1_idxs = split_idxs[split1]
                split2_idxs = split_idxs[split2]
                if split1 != split2:
                    split1_mask = ~np.all(
                        np.isin(triplets, split1_idxs), axis=1)
                    split2_mask = ~np.all(
                        np.isin(triplets, split2_idxs), axis=1)
                    combo_mask = np.logical_and(split1_mask, split2_mask)
                else:
                    combo_mask = np.all(np.isin(triplets, split1_idxs), axis=1)
                self.__log.debug('t_%s_%s %s' %
                                 (split1, split2,
                                  str(triplets[combo_mask].shape)))
                fh.create_dataset('t_%s_%s' % (split1, split2),
                                  data=triplets[combo_mask])
                fh.create_dataset('y_%s_%s' % (split1, split2),
                                  data=np.zeros((len(triplets[combo_mask]), )))
        self.__log.info('Triplets saved to %s', self.out_file)


@logged
class PrecomputedTripletSampler(BaseTripletSampler):
    """The triplets are not sampled but pre-computed."""

    def generate_triplets(self, X, ink_keys, triplets, out_file,
                          mean_center_x=True,
                          shuffle=True,
                          split_names=['train', 'test'],
                          split_fractions=[.8, .2],
                          suffix='eval', cpu=1,
                          x_dtype=float, y_dtype=float):

        try:
            from chemicalchecker.core.signature_data import DataSignature
        except ImportError as err:
            raise err

        # mean centering columns
        if mean_center_x:
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            if suffix is None:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler.pkl')
            else:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler_%s.pkl' % suffix)
            pickle.dump(scaler, open(scaler_file, 'wb'))

        # shuffling
        shuffle_idxs = np.arange(triplets.shape[0])
        if shuffle:
            np.random.shuffle(shuffle_idxs)
        triplets = np.array(triplets)[shuffle_idxs]

        # do train-test split on keys
        split_idxs = self.get_split_indeces(
            X.shape[0], split_fractions)

        # do train-test split for triplets (np.unique of indexes)
        split_idxs = dict(zip(split_names, split_idxs))

        # find triplets having test-test train-train and train-test
        combos = itertools.combinations_with_replacement(split_names, 2)

        # reverse split names to first write test keys (on a copy, so the
        # caller's list and the default argument are left untouched)
        split_names = list(reversed(split_names))

        # create dataset
        self.__log.info('Saving Triplets to %s', out_file)
        with h5py.File(out_file, "w") as fh:
            for split_n in split_names:
                fh.create_dataset('keys_%s' % split_n,
                                  data=np.array(ink_keys[split_idxs[split_n]],
                                                dtype=h5py.string_dtype()))
            if mean_center_x:
                fh.create_dataset(
                    'scaler',
                    data=np.array([scaler_file],
                                  dtype=h5py.string_dtype()))
            fh.create_dataset('x', data=X, dtype=x_dtype)
            fh.create_dataset('x_ink',
                              data=np.array(ink_keys,
                                            dtype=h5py.string_dtype()))

            for split1, split2 in combos:
                split1_idxs = split_idxs[split1]
                split2_idxs = split_idxs[split2]
                if split1 != split2:
                    split1_mask = ~np.all(
                        np.isin(triplets, split1_idxs), axis=1)
                    split2_mask = ~np.all(
                        np.isin(triplets, split2_idxs), axis=1)
                    combo_mask = np.logical_and(split1_mask, split2_mask)
                else:
                    combo_mask = np.all(np.isin(triplets, split1_idxs), axis=1)
                self.__log.debug('t_%s_%s %s' %
                                 (split1, split2,
                                  str(triplets[combo_mask].shape)))
                fh.create_dataset('t_%s_%s' % (split1, split2),
                                  data=triplets[combo_mask])
                fh.create_dataset('y_%s_%s' % (split1, split2),
                                  data=np.zeros((len(triplets[combo_mask]), )))
        self.__log.info('Triplets saved to %s', out_file)


@logged
class BinJaccardTripletSampler(BaseTripletSampler):
    """Triplets from the Jaccard similarity of binary profiles.

    Suited to small spaces: positives are each molecule's most similar
    molecules in a binary triplet signature (e.g. sign0 of a binary space),
    drawn with probability proportional to similarity. Formerly
    ``AdriaTripletSampler``, which remains as an alias.
    """

    def __init__(self, *args, **kwargs):
        BaseTripletSampler.__init__(self, *args, **kwargs)

    def generate_triplets(self, num_triplets=1e6, frac_hard=0.3,
                          frac_neig=0.05, metric='jaccard', low_thr=0.1,
                          high_thr=0.5, low_percentile=None,
                          high_percentile=None, seed=None, plot=True):
        """Generate triplets.

        Positives of each anchor are its most similar molecules in the binary
        triplet signature (similarity >= ``high_thr``, or at least its top
        ``frac_neig`` molecules with similarity >= ``low_thr``), drawn with
        probability proportional to similarity; negatives are the remaining
        molecules. ``frac_hard`` of the triplets are hard: positives from the
        top quartile of the anchor's neighbours, negatives from the lower
        quartiles.

        Molecules without any active feature and molecules whose closest
        neighbour is below ``low_thr`` get no triplets; both are logged.

        Args:
            num_triplets(int): Total number of triplets to generate.
            frac_hard(float): Fraction of triplets to be of the hard case.
            frac_neig(float): Fraction of neighbor we will consider.
            metric(str): Metric to compute similarities, must be a distance
                metric for binary data that can be converted to similarity
                by (1-dist).
            low_thr(float): Low similarity threshold, any pair below this is
                negative.
            high_thr(float): High similarity threshold, any pair above this is
                positive.
            low_percentile(float): If given, overrides ``low_thr`` with this
                percentile of all pairwise similarities (e.g. 60).
            high_percentile(float): If given, overrides ``high_thr`` with this
                percentile of all pairwise similarities (e.g. 99).
            seed(int): Seed for reproducible triplets.
            plot(bool): Save plots of the sampling.

        Raises:
            ValueError: If the triplet signature is not binary or lacks some
                of the molecules to represent.
        """
        self.__log.info('Generating Triplets...')
        self.__log.info('Triplets generated based on: %s' %
                        self.triplet_signature.data_path)
        self.__log.info('Triplets representation: %s' %
                        self.mol_signature.data_path)
        rng = np.random.default_rng(seed)
        mol_keys = np.array(self.mol_signature.keys)
        df = self.triplet_signature.as_dataframe()
        missing = np.setdiff1d(mol_keys, df.index.values)
        if len(missing) > 0:
            raise ValueError(
                '%i molecules to represent are missing from the triplet '
                'signature %s (e.g. %s)' % (len(missing),
                                            self.triplet_signature.data_path,
                                            missing[:3]))
        # molecules are represented by another signature (e.g. sign2),
        # we need to use only those molecules
        df = df.loc[mol_keys]
        values = df.values
        if not np.isin(values, [0, 1]).all():
            raise ValueError(
                'BinJaccardTripletSampler needs a binary triplet signature, but '
                '%s has values in [%.3g, %.3g]; pass a binary signature '
                '(e.g. sign0 of a binary space) as triplet_sign to '
                'sign3.fit, or binarize the data first' % (
                    self.triplet_signature.data_path,
                    np.nanmin(values), np.nanmax(values)))
        values = values.astype(bool)

        # molecules without any active feature are identical to each other
        # under jaccard, so they are left out
        active = values.any(axis=1)
        self.__log.info('Molecules without active features (no triplets): '
                        '%i of %i' % (np.sum(~active), len(active)))
        values = values[active]
        keys = mol_keys[active]

        # getting similarities
        all_similarities = 1 - pdist(values, metric)
        similarity = squareform(all_similarities).astype(np.float32)
        np.fill_diagonal(similarity, np.nan)  # masking itself
        if low_percentile is not None:
            low_thr = np.percentile(all_similarities, low_percentile)
        if high_percentile is not None:
            high_thr = np.percentile(all_similarities, high_percentile)
        self.__log.info('Similarity thresholds: low %.3f, high %.3f' %
                        (low_thr, high_thr))

        # molecules whose closest neighbour is below low_thr get no triplets
        closest = np.nanmax(similarity, axis=1)
        kept = closest >= low_thr
        self.__log.info('Molecules with closest neighbour below low_thr '
                        '(no triplets): %i' % np.sum(~kept))
        similarity = similarity[np.ix_(kept, kept)]
        keys = keys[kept]
        n_mols = len(keys)
        if n_mols < 3:
            raise ValueError('Only %i molecules left to sample triplets '
                             'from; lower low_thr' % n_mols)

        # defining derived parameters
        n_neigh = int(n_mols * frac_neig)
        n_trip = int(np.round(num_triplets * (1 - frac_hard) / n_mols))
        n_hard_trip = int(np.round(num_triplets * frac_hard / n_mols))

        triplets = []
        hard_triplets = {0: [], 1: [], 2: []}
        all_idxs = np.arange(n_mols)
        for anchor in tqdm(range(n_mols), total=n_mols):
            v = similarity[anchor]
            # getting positives
            pos_mask = v >= high_thr
            if np.sum(pos_mask) < n_neigh:
                candidates = np.where(v >= low_thr)[0]
                if len(candidates) == 0:
                    continue
                ranked = np.sort(v[candidates])[::-1]
                cutoff = ranked[min(n_neigh - 1, len(ranked) - 1)]
                pos_mask = v >= cutoff
            neighs = np.where(pos_mask)[0]
            similarities = v[neighs]
            probs = similarities / np.sum(similarities)
            # getting negatives: everything else but the anchor itself
            neg_mask = ~pos_mask
            neg_mask[anchor] = False
            negs = all_idxs[neg_mask]
            if len(negs) == 0:
                continue

            # easy triplets
            triplets.append(np.column_stack([
                np.full(n_trip, anchor),
                rng.choice(neighs, size=n_trip, p=probs),
                rng.choice(negs, size=n_trip)]))

            # hard triplets: positives from the top quartile of the
            # neighbours, negatives from each lower quartile
            scores = np.unique(similarities)
            cutoffs = np.unique([np.percentile(scores, pc)
                                 for pc in [0, 25, 50, 75, 100]])
            labels = np.arange(len(cutoffs))[:-1]
            groups = np.array(pd.cut(similarities, cutoffs, labels=labels))
            labels = [x for x in labels if x in groups]
            if len(labels) > 1:
                n_subhard = int(np.ceil(n_hard_trip / (len(labels) - 1)))
                top = groups == labels[-1]
                hard_positives = neighs[top]
                hard_probs = similarities[top] / np.sum(similarities[top])
                for label in labels[:-1]:
                    hard_negatives = neighs[groups == label]
                    hard_triplets[label].append(np.column_stack([
                        np.full(n_subhard, anchor),
                        rng.choice(hard_positives, size=n_subhard,
                                   p=hard_probs),
                        rng.choice(hard_negatives, size=n_subhard)]))

        empty = np.empty((0, 3), dtype=int)
        triplets = np.vstack(triplets) if triplets else empty
        hard_triplets = {g: np.vstack(t) if t else empty
                         for g, t in hard_triplets.items()}
        all_triplets = np.vstack([triplets] + list(hard_triplets.values()))
        if len(all_triplets) == 0:
            raise ValueError('No triplets could be sampled; check the '
                             'thresholds and the triplet signature')

        self.__log.info('triplets:      %i' % len(all_triplets))
        self.__log.info('easy triplets: %i (%.2f%%)' %
                        (len(triplets),
                         (100 * (len(triplets)) / len(all_triplets))))
        total_hard = np.sum([len(t) for t in hard_triplets.values()])
        self.__log.info('hard triplets: %i (%.2f%%)' %
                        (total_hard, (100 * (total_hard / len(all_triplets)))))
        for g in hard_triplets:
            self.__log.info('\t--> Q%i vs Q4: %i (%.2f%%)' %
                            (g + 1, len(hard_triplets[g]),
                             100 * len(hard_triplets[g]) / len(all_triplets)))

        # from positions among sampled molecules to rows of mol_signature
        mol_pos = dict(zip(mol_keys, np.arange(len(mol_keys))))
        to_mol_idx = np.array([mol_pos[k] for k in keys])
        all_triplets_idxs = to_mol_idx[all_triplets]
        all_triplets_idxs = all_triplets_idxs[
            np.lexsort(all_triplets_idxs.T[::-1])]

        if plot:
            self._plot_triplets(values, all_similarities, similarity,
                                closest, triplets, hard_triplets, low_thr,
                                high_thr, frac_neig, kept)

        self.save_triplets(all_triplets_idxs, **self.save_kwargs)

    def _plot_triplets(self, values, all_similarities, similarity, closest,
                       triplets, hard_triplets, low_thr, high_thr, frac_neig,
                       kept):
        """Save diagnostic plots of the sampling next to the output file."""
        self.__log.info('Generating Triplets Plot...')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import pearsonr
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=100)
        ax1, ax2, ax3, ax4, ax5, ax6 = axes.flat

        # which fraction of mols has a given # of features?
        n_features = values.sum(1)
        sns.ecdfplot(n_features, ax=ax1)
        ax1.set_title('# Features distribution')

        # which similarity will we always consider as positive or negative?
        ax2.set_title('Similarity distribution, pos./neg. definition')
        pc = int((1 - frac_neig) * 100)
        pc_val = np.percentile(all_similarities, pc)
        ax2.axvline(low_thr, label='low_thr %.2f' % low_thr, ls='-.',
                    color='.5')
        ax2.axvline(pc_val, label='%.2f (P%i)' % (pc_val, pc), color='.7')
        ax2.axvline(high_thr, label='high_thr %.2f' % high_thr, ls='--',
                    color='.5')
        sns.histplot(all_similarities, kde=True, ax=ax2)
        ax2.legend()

        # what fraction of mols do we lose (closest neigh < low_thr)?
        sns.ecdfplot(closest, ax=ax3)
        ax3.set_xlabel('Similarity to closest neighbor')
        ax3.axvline(low_thr, ls='-.', color='.5')
        ax3.axvline(high_thr, ls='--', color='.5')
        ax3.set_title('Closest neighbor of each mol. (lost: %i)' %
                      np.sum(~kept))

        # what's the difference in similarity between A-P and A-N?
        def deltas(t):
            return (similarity[t[:, 0], t[:, 1]] -
                    similarity[t[:, 0], t[:, 2]])
        ax4.set_title('Triplet difficulty and Anchor-Pos. vs. Anchor-Neg.')
        sns.ecdfplot(deltas(triplets), label='easy triplets', ax=ax4)
        for g, t in hard_triplets.items():
            if len(t):
                sns.ecdfplot(deltas(t), ax=ax4,
                             label='hard triplets (Q%i vs Q4)' % (g + 1))
        ax4.set_xlabel('pos.-neg. similarity delta')
        ax4.legend()

        # how often is each molecule used, and is it biased by # features?
        all_t = np.vstack([triplets] + list(hard_triplets.values()))
        n_kept = similarity.shape[0]
        usage = {role: np.bincount(all_t[:, i], minlength=n_kept)
                 for i, role in enumerate(['anchor', 'positive', 'negative'])}
        for role, counts in usage.items():
            sns.ecdfplot(counts, label=role, ax=ax5)
        ax5.set_xlabel('# triplets per molecule')
        ax5.set_title('Molecule usage per role')
        ax5.legend()

        kept_features = n_features[kept]
        r = pearsonr(kept_features, usage['positive'])[0]
        ax6.scatter(kept_features, usage['positive'], s=8, alpha=.5)
        ax6.set_xlabel('# active features')
        ax6.set_ylabel('# times used as positive')
        ax6.set_title('Feature-count bias (Pearson r %.2f)' % r)
        fig.tight_layout()
        plt.savefig(self.out_file + '.png')
        plt.close(fig)


# former name, kept so existing code keeps working
AdriaTripletSampler = BinJaccardTripletSampler


@logged
class OldTripletSampler(BaseTripletSampler):
    """Used to be the monstrous NeighborTripletTraintest. 
    Performs well on large spaces, less well on smaller ones"""

    def __init__(self, *args, **kwargs):
        BaseTripletSampler.__init__(self, *args, **kwargs)

    def generate_triplets(self, f_per=0.1, t_per=0.01,
                          mean_center_x=True, shuffle=True,
                          check_distances=True,
                          split_names=['train', 'test'],
                          split_fractions=[.8, .2],
                          suffix='eval', x_dtype=float,
                          y_dtype=float,
                          num_triplets=1e6, limit=100000, cpu=1):
        """Sample triplets using an approach suited for large spaces.

        Args:
            num_triplets(int): Total number of triplets to generate.
        """
        try:
            import faiss
            from chemicalchecker.core.signature_data import DataSignature
        except ImportError as err:
            raise err
        faiss.omp_set_num_threads(cpu)

        neighbors_sign = self.triplet_signature
        out_file = self.out_file
        ink_keys = self.mol_signature.keys
        _, X = self.mol_signature.get_vectors(ink_keys, dataset_name='x')

        # train test validation splits
        if len(split_names) != len(split_fractions):
            raise Exception("Split names and fraction should be same amount.")

        # Load neigh matrix and shuffle it
        neighbors_matrix = neighbors_sign[:]
        shuffle_idx = np.arange(neighbors_matrix.shape[0])
        np.random.shuffle(shuffle_idx)

        OldTripletSampler.__log.debug('%s %s' % (
            len(neighbors_matrix), str(X.shape)))
        if len(neighbors_matrix) != X.shape[0]:
            raise Exception("neighbors_matrix should be same length as X.")

        neighbors_matrix = neighbors_matrix[shuffle_idx]
        X = X[shuffle_idx]
        X_inks = np.array(neighbors_sign.keys)[shuffle_idx]

        OldTripletSampler.__log.debug(
            "{:<20} shape: {:>10}".format("input X", str(X.shape)))

        fullpath, _ = os.path.split(out_file)
        redundancy_path = os.path.join(fullpath, "redundancy_dict.pkl")
        # reduce redundancy, keep full-ref mapping
        # if not os.path.isfile(redundancy_path):
        OldTripletSampler.__log.info("Reducing redundancy")
        rnd = RNDuplicates(cpu=cpu)
        _, ref_matrix, full_ref_map = rnd.remove(
            neighbors_matrix.astype(float))
        ref_full_map = dict()
        for key, value in full_ref_map.items():
            ref_full_map.setdefault(value, list()).append(key)
        full_refid_map = dict(
            zip(rnd.final_ids, np.arange(len(rnd.final_ids))))
        refid_full_map = {full_refid_map[k]: v
                          for k, v in ref_full_map.items()}

        # Limit signatures by limit value
        size_original_ref_matrix = len(ref_matrix)
        OldTripletSampler.__log.info(
            "Original size ref_matrix: %s" % size_original_ref_matrix)
        OldTripletSampler.__log.info("Limit of %s" % limit)
        ref_matrix = ref_matrix[:limit]
        OldTripletSampler.__log.info("Final size: %s" % len(ref_matrix))

        # Set triplet_factors
        triplet_per_mol = max(
            [int(np.ceil(num_triplets / ref_matrix.shape[0])), 3])
        easy_triplet_per_mol = max([int(np.ceil(triplet_per_mol * 0.8)), 1])
        # triplet_per_mol - (2 * easy_triplet_per_mol)
        medi_triplet_per_mol = max([int(np.ceil(triplet_per_mol * 0.15)), 1])
        # easy_triplet_per_mol
        hard_triplet_per_mol = max(
            [triplet_per_mol -
             (easy_triplet_per_mol + medi_triplet_per_mol), 1])
        OldTripletSampler.__log.info(
            "Triplet_per_mol: %s" % triplet_per_mol)
        OldTripletSampler.__log.info(
            "E triplet per mol: %s" % easy_triplet_per_mol)
        OldTripletSampler.__log.info(
            "M triplet per mol: %s" % medi_triplet_per_mol)
        OldTripletSampler.__log.info(
            "H triplet per mol: %s" % hard_triplet_per_mol)
        OldTripletSampler.__log.info("Triplet_per_mol: %s" % (
            easy_triplet_per_mol + medi_triplet_per_mol +
            hard_triplet_per_mol))
        assert(triplet_per_mol <= (easy_triplet_per_mol +
                                   medi_triplet_per_mol +
                                   hard_triplet_per_mol))

        # split chunks, get indexes of chunks for each split
        chunk_size = int(max(1, np.floor(ref_matrix.shape[0] / 100)))
        split_chunk_idx = self.get_split_indeces(
            int(np.floor(ref_matrix.shape[0] / chunk_size)) + 1,
            split_fractions)
        tot_split = float(sum([len(i) for i in split_chunk_idx]))
        real_fracs = ['%.2f' % (len(i) / tot_split) for i in split_chunk_idx]
        OldTripletSampler.__log.info(
            'Fractions used: %s', ' '.join(real_fracs))

        # split ref matrix, keep ref-split mapping
        nr_matrix = dict()
        split_ref_map = dict()
        for split_name, chunks in zip(split_names, split_chunk_idx):
            # need total size and mapping of chunks
            src_dst = list()
            total_size = 0
            for dst, src in enumerate(sorted(chunks)):
                # source chunk start-end
                src_start = src * chunk_size
                src_end = (src * chunk_size) + chunk_size
                # check current chunk size to avoid overflowing
                curr_chunk_size = chunk_size
                if src_end > ref_matrix.shape[0]:
                    src_end = ref_matrix.shape[0]
                    curr_chunk_size = src_end - src_start
                # update total size
                total_size += curr_chunk_size
                # destination start-end
                dst_start = dst * chunk_size
                dst_end = (dst * chunk_size) + curr_chunk_size
                src_slice = (int(src_start), int(src_end))
                dst_slice = (int(dst_start), int(dst_end))
                src_dst.append((src_slice, dst_slice))
            # create chunk matrix
            cols = ref_matrix.shape[1]
            nr_matrix[split_name] = np.zeros((int(total_size), cols),
                                             dtype=ref_matrix.dtype)
            split_ref_map[split_name] = dict()
            ref_idxs = np.arange(ref_matrix.shape[0])
            for src_slice, dst_slice in tqdm(src_dst):
                src_chunk = slice(*src_slice)
                dst_chunk = slice(*dst_slice)
                OldTripletSampler.__log.debug(
                    "writing src: %s  to dst: %s" % (src_slice, dst_slice))
                ref_src_chunk = ref_idxs[src_chunk]
                ref_dst_chunk = ref_idxs[dst_chunk]
                for src_id, dst_id in zip(ref_src_chunk, ref_dst_chunk):
                    split_ref_map[split_name][dst_id] = src_id
                nr_matrix[split_name][dst_chunk] = ref_matrix[src_chunk]
            OldTripletSampler.__log.debug(
                "nr_matrix %s %s", split_name, nr_matrix[split_name].shape)

        # for each split generate NN
        OldTripletSampler.__log.info('Generating NN indexes')
        NN = dict()
        for split_name in split_names:
            # create faiss index
            NN[split_name] = faiss.IndexFlatL2(nr_matrix[split_name].shape[1])
            # add data
            NN[split_name].add( np.array(nr_matrix[split_name], dtype='float32') )

        # mean centering columns
        if mean_center_x:
            OldTripletSampler.__log.info('Mean centering X')
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
            if suffix is None:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler.pkl')
            else:
                scaler_file = os.path.join(os.path.split(out_file)[0],
                                           'scaler_%s.pkl' % suffix)
            pickle.dump(scaler, open(scaler_file, 'wb'))

        # create dataset
        OldTripletSampler.__log.info('Traintest saving to %s', out_file)
        combo_dists = dict()
        with h5py.File(out_file, "w") as fh:
            fh.create_dataset('x', data=X)
            fh.create_dataset('x_ink', data=np.array(
                X_inks, dtype=DataSignature.string_dtype()))
            if mean_center_x:
                fh.create_dataset(
                    'scaler',
                    data=np.array([scaler_file],
                                  dtype=DataSignature.string_dtype()))
            # for each split combo generate triplets where [anchor, positive,
            # negative]
            combos = itertools.combinations_with_replacement(split_names, 2)
            for split1, split2 in combos:
                combo = '_'.join([split1, split2])
                OldTripletSampler.__log.debug("SPLIT: %s" % combo)
                # define F and T according to the split that is being used

                LB = 10000
                UB = 100000
                TMAX = 50
                TMIN = 5

                def get_t_max(N):
                    N = np.clip(N, LB, UB)
                    a = (TMAX - TMIN) / (LB - UB)
                    b = TMIN - a * UB
                    return int(a * N + b)

                t_limit = get_t_max(size_original_ref_matrix)
                f_limit = 300

                T = int(
                    np.clip(t_per * nr_matrix[split2].shape[0], 10, t_limit))
                F = np.clip(10 * T, 200, f_limit)
                F = int(min(F, (nr_matrix[split2].shape[0] - 1)))

                OldTripletSampler.__log.info("T per: %s" % (t_per))
                OldTripletSampler.__log.info("F and T: %s %s" % (F, T))
                assert(T < F)

                OldTripletSampler.__log.info("Searching Neighbors")
                # remove self neighbors when splits are the same
                if split1 == split2:
                    # search NN in chunks
                    neig_idxs = list()
                    csize = 10000
                    for i in tqdm(range(0, len(nr_matrix[split2]), csize)):
                        chunk = slice(i, i + csize)
                        _, neig_idxs_chunk = NN[split1].search(
                            np.array( nr_matrix[split2][chunk], dtype='float32'), F + 1)
                        neig_idxs.append(neig_idxs_chunk)
                    neig_idxs = np.vstack(neig_idxs)
                    # the nearest neig between same groups is the molecule
                    # itself
                    # assert(all(neig_idxs[:, 0] ==
                    #           np.arange(0, len(neig_idxs))))
                    neig_idxs = neig_idxs[:, 1:]
                else:
                    _, neig_idxs = NN[split1].search(
                        np.array( nr_matrix[split2], dtype='float32'), F)

                # get probabilities for T
                t_prob = ((np.arange(T + 1)[::-1]) /
                          np.sum(np.arange(T + 1)))[:T]
                assert(sum(t_prob) > 0.99)

                # save list of split indeces

                # anchors_split = np.repeat(np.arange(len(neig_idxs)),
                # triplet_per_mol)
                easy_a_split = list()
                easy_p_split = list()
                easy_n_split = list()
                medi_a_split = list()
                medi_p_split = list()
                medi_n_split = list()
                hard_a_split = list()
                hard_p_split = list()
                hard_n_split = list()

                OldTripletSampler.__log.info("Generating triplets")
                nn_set = set(range(neig_idxs.shape[0]))

                # idx refere split2, all else to split1
                for idx, row in enumerate(tqdm(neig_idxs)):
                    # Add acnhors per type of triplet
                    easy_a_split.extend(np.repeat(idx, easy_triplet_per_mol))
                    medi_a_split.extend(np.repeat(idx, medi_triplet_per_mol))
                    hard_a_split.extend(np.repeat(idx, hard_triplet_per_mol))

                    # positives are samples from tot T NNs for each category
                    #    Easy
                    e_p_indexes = np.random.choice(
                        T, easy_triplet_per_mol, replace=True, p=t_prob)
                    positives = neig_idxs[idx, e_p_indexes]
                    easy_p_split.extend(positives)

                    #    Medium
                    m_p_indexes = np.random.choice(
                        T, medi_triplet_per_mol, replace=True, p=t_prob)
                    positives = neig_idxs[idx, m_p_indexes]
                    medi_p_split.extend(positives)

                    #    Hard
                    h_p_indexes = np.random.choice(
                        T, hard_triplet_per_mol, replace=True, p=t_prob)
                    positives = neig_idxs[idx, h_p_indexes]
                    hard_p_split.extend(positives)

                    """
                    p_indexes = np.random.choice(T, triplet_per_mol, 
                    replace=True, p=t_prob)
                    positives = neig_idxs[idx, p_indexes]
                    easy_p_split.extend(positives)
                    medi_p_split.extend(positives)
                    hard_p_split.extend(positives)"""

                    # medium negatives are sampled from F (in NN but not T)
                    m_negatives = np.random.choice(
                        neig_idxs[idx][T:], medi_triplet_per_mol, replace=True)
                    medi_n_split.extend(m_negatives)

                    # hard negatives are sampled from T (but higher than
                    # positives)
                    hn_shifts = np.random.choice(
                        int(np.ceil(T / 2)), hard_triplet_per_mol,
                        replace=True) + 1
                    hn_indexes = hn_shifts + h_p_indexes
                    # with small T we still have to avoid getting out of T
                    # range
                    off_range = np.where(hn_indexes >= neig_idxs.shape[1])
                    hn_indexes[off_range] = neig_idxs.shape[1] - 1
                    h_negatives = neig_idxs[idx, hn_indexes]
                    hard_n_split.extend(h_negatives)

                    # easy negatives (sampled from everywhere; in general
                    # should be fine altough it may sample positives...)
                    e_negatives = np.random.choice(
                        len(neig_idxs), easy_triplet_per_mol, replace=True)
                    easy_n_split.extend(e_negatives)

                # get reference ids
                OldTripletSampler.__log.info("Mapping triplets")
                # anchors_ref = [split_ref_map[split2][x] for x in
                # anchors_split]
                easy_a_ref = [split_ref_map[split2][x] for x in easy_a_split]
                easy_p_ref = [split_ref_map[split1][x] for x in easy_p_split]
                easy_n_ref = [split_ref_map[split1][x] for x in easy_n_split]
                medi_a_ref = [split_ref_map[split2][x] for x in medi_a_split]
                medi_p_ref = [split_ref_map[split1][x] for x in medi_p_split]
                medi_n_ref = [split_ref_map[split1][x] for x in medi_n_split]
                hard_a_ref = [split_ref_map[split2][x] for x in hard_a_split]
                hard_p_ref = [split_ref_map[split1][x] for x in hard_p_split]
                hard_n_ref = [split_ref_map[split1][x] for x in hard_n_split]

                # choose random from full analogs
                OldTripletSampler.__log.info(
                    "Resolving multiple options")
                easy_a_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in easy_a_ref])
                easy_p_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in easy_p_ref])
                easy_n_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in easy_n_ref])
                medi_a_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in medi_a_ref])
                medi_p_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in medi_p_ref])
                medi_n_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in medi_n_ref])
                hard_a_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in hard_a_ref])
                hard_p_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in hard_p_ref])
                hard_n_full = np.array(
                    [np.random.choice(refid_full_map[x]) for x in hard_n_ref])

                # stack triplets
                OldTripletSampler.__log.info("Stacking triplets")
                easy_triplets = np.vstack(
                    (easy_a_full, easy_p_full, easy_n_full)).T
                medium_triplets = np.vstack(
                    (medi_a_full, medi_p_full, medi_n_full)).T
                hard_triplets = np.vstack(
                    (hard_a_full, hard_p_full, hard_n_full)).T
                triplets = np.vstack(
                    (easy_triplets, medium_triplets, hard_triplets))
                # stack categories
                y = np.hstack((
                    np.full((easy_triplets.shape[0],), 0),
                    np.full((medium_triplets.shape[0],), 1),
                    np.full((hard_triplets.shape[0],), 2)))

                unique_ids = np.unique(triplets)
                OldTripletSampler.__log.info(
                    'Using %s molecules in triplets' %
                    len(unique_ids))

                # get inchikeys of test or train molecules
                if split1 == split2:
                    ink_ids = np.array(sorted(unique_ids))
                    split_inks = np.array(np.array(X_inks[ink_ids]),
                                          DataSignature.string_dtype())
                    ds_name = "keys_%s" % split1
                    fh.create_dataset(ds_name, data=split_inks)
                # save to h5
                ds_name = "t_%s_%s" % (split1, split2)
                ys_name = "y_%s_%s" % (split1, split2)
                _, unique_idx = np.unique(triplets, axis=0, return_index=True)
                # check for all categories to still be there
                if len(np.unique(y[unique_idx])) < 3:
                    OldTripletSampler.__log.warning(
                        'Very few molecules available... triplets will be '
                        'repeated in the difficulty categories.')
                    # this can happend when we have very few molecules
                    ty = np.hstack([triplets, np.expand_dims(y, 1)])
                    tripletsy = np.unique(ty, axis=0)
                    triplets = tripletsy[:, :3]
                    y = tripletsy[:, -1]
                else:
                    triplets = triplets[unique_idx]
                    y = y[unique_idx]
                # shuffling
                shuffle_idxs = np.arange(triplets.shape[0])
                if shuffle:
                    np.random.shuffle(shuffle_idxs)
                triplets = triplets[shuffle_idxs]
                y = y[shuffle_idxs]
                OldTripletSampler.__log.info(
                    'Using %s unique triplets' % len(y))
                OldTripletSampler.__log.info(
                    'writing Name: %s E: %s M: %s H: %s T: %s', ds_name, y[
                        y == 0].shape[0],
                    y[y == 1].shape[0], y[y == 2].shape[0], triplets.shape[0])
                fh.create_dataset(ds_name, data=triplets)
                fh.create_dataset(ys_name, data=y)

                if check_distances:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    num_of_dist_errors = 0
                    dis_limit = min(50000, len(shuffle_idxs))
                    dists = np.empty((dis_limit, 3))
                    for idx, row in enumerate(shuffle_idxs[:dis_limit]):
                        anchor = neighbors_matrix[triplets[row][0]]
                        positive = neighbors_matrix[triplets[row][1]]
                        negative = neighbors_matrix[triplets[row][2]]
                        category = y[row]

                        dis_ap = euclidean(anchor, positive)
                        dis_an = euclidean(anchor, negative)
                        dists[idx] = [dis_ap, dis_an, category]
                        if (dis_ap > dis_an):
                            # OldTripletSampler.__log.warning(
                            #    'DIST ERROR %s %.2f %.2f %i' %
                            #    (triplets[row], dis_ap, dis_an, category))
                            num_of_dist_errors += 1
                    OldTripletSampler.__log.warning(
                        'TOTAL DIST ERRORS:  %s' % num_of_dist_errors)
                    assert(len(np.unique(dists[:, 2])) == 3)
                    combo_dists[combo] = dists

        if check_distances:
            fig, axes = plt.subplots(
                3, 3, sharex=True, sharey=False, figsize=(10, 10))
            ax_idx = 0
            cat_names = ['easy', 'medium', 'hard']
            for combo, dists in combo_dists.items():
                for cat_id in [0, 1, 2]:
                    cat_mask = dists[:, 2] == cat_id
                    ax = axes.flatten()[ax_idx]
                    ax.set_title('%s %s' % (combo, cat_names[cat_id]))
                    sns.histplot(dists[cat_mask, 0], label='AP',
                                 color='green', kde=True, ax=ax)
                    sns.histplot(dists[cat_mask, 1], label='AN',
                                 color='red', kde=True, ax=ax)
                    ax.legend()
                    ax_idx += 1

            if suffix is None:
                plot_file = os.path.join(
                    os.path.split(out_file)[0], 'distances.png')
            else:
                plot_file = os.path.join(
                    os.path.split(out_file)[0], 'distances_%s.png' % suffix)
            plt.savefig(plot_file)
            plt.close()

        OldTripletSampler.__log.info(
            'OldTripletSampler saved to %s', out_file)
