"""Diagnostics for Signatures.

Performs an array of validations and diagnostic analysis.
"""
import os
import random
import pickle
import shutil
import collections
import numpy as np
import tempfile
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import normalize
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from chemicalchecker.util import logged, Config
from chemicalchecker.util.plot import DiagnosisPlot
#from chemicalchecker.util.decorator import safe_return
from chemicalchecker.util.hpc import HPC


@logged
class Diagnosis(object):
    """Diagnosis class."""

    def __init__(self, sign, ref_cc=None, ref_cctype=None, save=True,
                 plot=True, overwrite=False, load=True, n=10000, seed=42,
                 cpu=4):
        """Initialize a Diagnosis instance.

        Args:
            ref_cc (ChemicalChecker): A CC instance used as reference.
            sign (CC signature): The CC signature object to be diagnosed.
            save (bool): Whether to save results in the `diags` folder of the
                signature. (default=True)
            plot (bool): Whether to save plots in the `diags` folder of the
                signature. (default=True)
            overwrite (bool): Whether to overwrite the results of the
                diagnosis. (default=False)
            n (int): Number of molecules to sample. (default=10000)
        """
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        if ref_cc is None:
            ref_cc = sign.get_cc()
        if ref_cctype is None:
            ref_cctype = sign.cctype
        self.ref_cc = ref_cc
        self.save = save
        self.plot = plot
        self.plotter = DiagnosisPlot(self)
        self.sign = sign
        self.ref_cctype = ref_cctype
        self.subsample_n = n
        self.cpu = cpu
        self.memory = 5
        # check if reference CC has reference all cctype signatures
        
        # define current diag_path
        self.name = '%s_%s' % (ref_cc.name, self.ref_cctype)
        self.path = os.path.join(sign.diags_path, self.name)
        if os.path.isdir(self.path):
            if overwrite:
                self.clear()
        if not os.path.isdir(self.path):
            os.mkdir(self.path)
        if self.plot and not self.save:
            self.__log.warning(
                "Saving is necessary to plot. Setting 'save' to True.")
            self.save = True
        self._V = None
        self._keys = None

    @property
    def V(self):
        if self._V is None:
            self._V, self._keys = self._subsample()
        return self._V

    @property
    def keys(self):
        if self._keys is None:
            self._V, self._keys = self._subsample()
        return self._keys

    def _subsample(self):
        fn = os.path.join(self.path, "subsampled_data.pkl")
        if not os.path.exists(fn):
            self.__log.debug("Subsampling signature")
            V, keys = self.sign.subsample(self.subsample_n, seed=self.seed)
            if self.save:
                self.__log.debug("Saving signature subsample %s" % fn)
                with open(fn, "wb") as fh:
                    d = {"V": V, "keys": keys}
                    pickle.dump(d, fh)
            return V, keys
        self.__log.debug("Loading signature subsample")
        with open(fn, "rb") as f:
            d = pickle.load(f)
        return d["V"], d["keys"]

    def clear(self):
        """Remove al diagnostic data."""
        self.__log.debug("Deleting %s" % self.path)
        shutil.rmtree(self.path)

    def _todo(self, fn, inner=False):
        """Check if function is to be run."""
        return not os.path.exists(os.path.join(self.path, fn + ".pkl"))

    def _load_diagnosis_pickle(self, fn):
        with open(os.path.join(self.path, fn), "rb") as f:
            results = pickle.load(f)
        return results

    def _returner(self, *args, results, fn, save, plot, plotter_function,
                  kw_plotter=dict(), **kwargs):
        fn_ = os.path.join(self.path, fn + ".pkl")
        if results is None:
            results = pickle.load(open(fn_, "rb"))
        if save:
            pickle.dump(results, open(fn_, "wb"))
            if plot:
                return plotter_function(results=results, **kw_plotter)
            else:
                return fn_
        else:
            if plot:
                return plotter_function(results=results, **kw_plotter)
            else:
                return results

    def _shared_keys(self, sign):
        return sorted(list(set(self.keys) & set(sign.keys)))

    def _paired_keys(self, my_keys, vs_keys):
        keys = sorted(set(my_keys) & set(vs_keys))
        my_keys = keys
        vs_keys = keys
        return np.array(my_keys), np.array(vs_keys)

    def _paired_conn_layers(self, my_keys, vs_keys):
        vs_conn_set = set([ik.split("-")[0] for ik in vs_keys])
        my_conn_set = set([ik.split("-")[0] for ik in my_keys])
        common_conn = my_conn_set & vs_conn_set
        vs_keys_conn = dict(
            (ik.split("-")[0], ik) for ik in vs_keys if ik.split("-")[0] in common_conn)
        my_keys_conn = dict(
            (ik.split("-")[0], ik) for ik in my_keys if ik.split("-")[0] in common_conn)
        common_conn = sorted(common_conn)
        my_keys = [my_keys_conn[c] for c in common_conn]
        vs_keys = [vs_keys_conn[c] for c in common_conn]
        return np.array(my_keys), np.array(vs_keys)

    def _select_datasets(self, datasets, exemplary):
        if datasets is None:
            datasets = self.ref_cc.datasets
        dss = []
        for ds in datasets:
            if exemplary:
                # TODO: to change in case of CC expansion from 25 spaces
                if ds[0] in "ABCDE" and ds[1] in "12345" and ds[-3:] == "001":
                    dss += [ds]
            else:
                dss += [ds]
        return dss

    def _get_signatures(self, *args, keys=None, max_keys=10000,
                        max_features=None, shuffle=False, **kwargs):
        if max_keys is None:
            max_keys = len(self.keys)
        if keys is None:
            V = self.V
            keys = np.array(self.keys)
            if shuffle:
                idxs = np.random.choice(len(keys), np.min(
                    [max_keys, len(keys)]), replace=False)
                V = V[idxs]
                keys = keys[idxs]
            else:
                V = V[:max_keys]
                keys = keys[:max_keys]
        else:
            keys = set(keys).intersection(self.sign.keys)
            self.__log.debug("%d keys found" % len(keys))
            keys = sorted(random.sample(keys, np.min([max_keys, len(keys)])))
            self.__log.debug("Fetching signatures")
            keys, V = self.sign.get_vectors(keys)
            if shuffle:
                idxs = np.random.choice(len(keys), np.min(
                    [max_keys, len(keys)]), replace=False)
                V = V[idxs]
                keys = keys[idxs]
            else:
                V = V[:max_keys]
                keys = keys[:max_keys]

        if max_features is not None and max_features < V.shape[1]:
            idxs = np.random.choice(V.shape[1], max_features, replace=False)
            V = V[:, idxs]

        return V, keys

    def _distance_distribution(self, metric, max_keys=200):
        """Distance distribution. Sampled with replacement.

        Args:
            metric (str): 'cosine' or 'euclidean'. (default='cosine')
            max_keys (int): maximum number of keys to use for pairwise distance
                calculation.
        """
        if metric != "cosine" and metric != "euclidean":
            raise Exception("metric needs to be 'cosine' or 'euclidean'")
        n = self.V.shape[0]
        idxs = np.random.choice(n, max_keys)
        dists = pdist(self.V[idxs], metric=metric)
        order = np.argsort(dists)
        results = {
            "dists": dists[order]
        }
        return results

    def canvas_hpc(self, tmpdir, **kwargs):
        """Run HPC jobs .

        tmpdir(str): Folder (usually in scratch) where the job directory is
            generated.
        cc_root: CC root path
        cctype:  CC type (sign0, sign1, sign2, sign3) on which the method is applied
        molset: 'full' or 'reference'
        dss: datasets to run the diagnostics on
        cc_reference: another version of CC to use as diagnostic reference
        """
        cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
        cfg = Config(cc_config)
        job_name = "_".join(["CC", self.sign.cctype.upper(), self.sign.dataset,
                             "DIAGNOSIS"])
        job_path = tempfile.mkdtemp("_" + job_name, dir=tmpdir)
        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        # in case some plots of the canvases must be skipped
        skip_plots = kwargs.get("skip_plots", [])
        # create script file
        script_lines = f"""
import sys, os
import pickle
from chemicalchecker import ChemicalChecker
from chemicalchecker.core.diagnostics import Diagnosis
ChemicalChecker.set_verbosity('DEBUG')
task_id = sys.argv[1]
filename = sys.argv[2]
diag = pickle.load(open(filename, 'rb'))[task_id][0]
diag.canvas(size='small', savefig=True, skip_plots={skip_plots})
diag.canvas(size='medium', savefig=True, skip_plots={skip_plots})
print('JOB DONE')
        """
        script_name = os.path.join(job_path, 'diagnostics_script.py')
        with open(script_name, 'w') as fh:
            fh.write(script_lines)
        # HPC parameters
        params = kwargs
        params["num_jobs"] = 1
        params["jobdir"] = job_path
        params["job_name"] = kwargs.get('job_name', job_name)
        params["elements"] = [self]
        params["wait"] = kwargs.get('wait', False)
        params["check_error"] = kwargs.get('check_error', False)
        params["cpu"] = kwargs.get('cpu', self.cpu)
        params["mem_by_core"] = kwargs.get('mem_by_core', 4)
        # job command
        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" \
            " singularity exec {} python {} <TASK_ID> <FILE>"
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(cfg)
        cluster.submitMultiJob(command, **params)
        return cluster

    @staticmethod
    def diagnostics_hpc(tmpdir, cc_root, cctype, molset, dss, cc_reference, **kwargs):
        """Run HPC jobs .

        tmpdir(str): Folder (usually in scratch) where the job directory is
            generated.
        cc_root: CC root path
        cctype:  CC type (sign0, sign1, sign2, sign3) on which the method is applied
        molset: 'full' or 'reference'
        dss: datasets to run the diagnostics on
        cc_reference: another version of CC to use as diagnostic reference
        """
        cc_config = kwargs.get("cc_config", os.environ['CC_CONFIG'])
        cfg = Config(cc_config)
        job_path = tempfile.mkdtemp(
            prefix='jobs_diagnostics_' + cctype + "_", dir=tmpdir)
        # create job directory if not available
        if not os.path.isdir(job_path):
            os.mkdir(job_path)
        dataset_codes = list()
        for ds in dss:
            dataset_codes.append(ds)
        sign_args_tmp = kwargs.get('sign_args', {})
        sign_kwargs_tmp = kwargs.get('sign_kwargs', {})
        dataset_params = list()
        for ds_code in dataset_codes:
            sign_args = list()
            sign_args.extend(sign_args_tmp.get(ds_code, list()))
            sign_kwargs = dict()
            sign_kwargs.update(sign_kwargs_tmp.get(ds_code, dict()))
            sign_args.insert(0, cctype)
            sign_args.insert(1, molset)
            sign_args.insert(2, ds_code)
            dataset_params.append(
                (sign_args, sign_kwargs))
        # create script file
        script_lines = [
            "import sys, os",
            "import pickle",
            "from chemicalchecker import ChemicalChecker",
            "from chemicalchecker.core.diagnostics import Diagnosis",
            "task_id = sys.argv[1]",
            "filename = sys.argv[2]",
            "inputs = pickle.load(open(filename, 'rb'))",
            "sign_args = inputs[task_id][0][0]",
            "sign_kwargs = inputs[task_id][0][1]",
            "cc = ChemicalChecker('{cc_root}')",
            "sign = cc.get_signature(*sign_args, **sign_kwargs)",
            "cc_ref = ChemicalChecker('{cc_reference}')",
            "diag = Diagnosis(sign, cc_ref)",
            "fig = diag.canvas()",
            "fig.savefig(os.path.join(sign.diags_path, diag.name + '.png'))",
            "print('JOB DONE')"
        ]
        replacements = {"cc_root"}
        if cc_reference == "":
            cc_reference = cc_root
        script_name = os.path.join(job_path, 'diagnostics_script.py')
        with open(script_name, 'w') as fh:
            for line in script_lines:
                fh.write(line.format(cc_root=cc_root,
                                     cc_reference=cc_reference) + '\n')
        # HPC parameters
        params = {}
        params["num_jobs"] = len(dataset_codes)
        params["jobdir"] = job_path
        params["job_name"] = "CC_" + cctype.upper() + "_DIAGNOSIS"
        params["elements"] = dataset_params
        params["wait"] = False
        params["check_error"] = False
        # NB: slurm.py no longer honors the plain "memory" kwarg (the --mem
        # block is commented out); memory is only requested through
        # --mem-per-cpu, i.e. via "mem_by_core" * "cpu". Without this the
        # jobs fall back to the partition default and OOM on the large
        # molecule spaces (B4/B5, C3/C4/C5). Total mem = cpu * mem_by_core.
        params["cpu"] = kwargs.get("cpu", 4)
        params["mem_by_core"] = kwargs.get("mem_by_core", 16)  # 4*16 = 64GB
        # job command
        singularity_image = cfg.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={}" \
            " singularity exec {} python {} <TASK_ID> <FILE>"
        command = command.format(
            os.path.join(cfg.PATH.CC_REPO, 'package'), cc_config,
            singularity_image, script_name)
        # submit jobs
        cluster = HPC.from_config(cfg)
        cluster.submitMultiJob(command, **params)
        return cluster

    # @safe_return(None)
    def euclidean_distances(self, *arg, n_pairs=10000, **kwargs):
        """Euclidean distance distribution.

        Args:
            n_pairs (int): Number of pairs to sample. (default=10000)
        """
        self.__log.debug("Euclidean distances")
        fn = "euclidean_distances"
        if self._todo(fn):
            results = self._distance_distribution(metric="euclidean")
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.euclidean_distances,
            **kwargs)

    # @safe_return(None)
    def cosine_distances(self, *args, n_pairs=10000, **kwargs):
        """Cosine distance distribution.

        Args:
            n_pairs (int): Number of pairs to sample. (default=10000)
        """
        self.__log.debug("Cosine distances")
        fn = "cosine_distances"
        if self._todo(fn):
            results = self._distance_distribution(metric="cosine")
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.cosine_distances,
            **kwargs)

    # @safe_return(None)
    def cross_coverage(self, dataset, *args, molset="full",
                       try_conn_layer=False, redo=False, **kwargs):
        """Intersection of coverages.

        Args:
            sign (signature): A CC signature object to check against.
        """
        
        ref_cctype = kwargs.get('ref_cctype', None)
        if ref_cctype is None:
            ref_cctype = 'sign1'
        
        qualified_name = '_'.join([dataset, ref_cctype, molset])
        fn = os.path.join(self.path, "cross_coverage_%s" % qualified_name)
        if self._todo(fn) or redo:
            metadata = self.ref_cc.sign_metadata(
                'keys', molset, dataset, ref_cctype)
            self.__log.debug("--cross cov %s, %s, %s" % (molset, dataset, ref_cctype) )
            
            if metadata is not None:
                vs_keys = metadata
            my_keys = self.sign.keys
            # apply inchikey connectivity layer if possible
            if try_conn_layer:
                keys1, keys2 = self._paired_conn_layers(my_keys, vs_keys)
            else:
                keys1, keys2 = self._paired_keys(my_keys, vs_keys)
            results = {
                "inter": len(keys1),
                "my_overlap": len(keys1) / len(my_keys),
                "vs_overlap": len(keys1) / len(vs_keys)
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.cross_coverage,
            kw_plotter={"sign_qualified_name": qualified_name},
            **kwargs)

    # @safe_return(None)
    def cross_roc(self, sign, *args, n_samples=10000, n_neighbors=5,
                  neg_pos_ratio=1, apply_mappings=False, try_conn_layer=False,
                  metric='cosine', redo=False, val_type='roc', **kwargs):
        """Perform validations.

        Args:
            sign (signature): A CC signature object to validate against.
            n_samples (int): Number of samples.
            apply_mappings (bool): Whether to use mappings to compute
                validation. Signature which have been redundancy-reduced
                (i.e. `reference`) have fewer molecules. The key are molecules
                from the `full` signature and values are molecules from the
                `reference` set.
            try_conn_layer (bool): Try with the inchikey connectivity layer.
                (default=False)
            metric (str): 'cosine' or 'euclidean'. (default='cosine')
            val_type (str): 'roc' or 'pr'. (default='roc')
            save (bool): Specific save parameter. If not specified, the global
                is set. (default=None).
        """
        fn = os.path.join(self.path, "cross_roc_%s" %
                          sign.qualified_name)
        if self._todo(fn) or redo:
            self.__log.debug("--cross %s, %s" % (sign.dataset, sign.cctype) )
            r = self.cross_coverage(sign.dataset, ref_cctype=sign.cctype,
                                    molset=sign.molset,
                                    apply_mappings=apply_mappings,
                                    try_conn_layer=try_conn_layer, save=False,
                                    redo=redo, plot=False)
            if r["inter"] < n_neighbors:
                self.__log.warning("Not enough shared molecules")
                return None
            my_keys = self.sign.keys
            vs_keys = sign.keys
            # apply inchikey connectivity layer if possible
            if try_conn_layer:
                keys1, keys2 = self._paired_conn_layers(my_keys, vs_keys)
            else:
                keys1, keys2 = self._paired_keys(my_keys, vs_keys)
            # Get fraction of shared samples
            if n_samples < len(keys1):
                idxs = sorted(np.random.choice(
                    len(keys1), n_samples, replace=False))
                keys1 = keys1[idxs]
                keys2 = keys2[idxs]
            # fit nearest neighbors on other signature
            vs_vectors = sign.get_vectors(keys2)[1]
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric,
                                  n_jobs=self.cpu)
            nn.fit(vs_vectors)
            # get positive pairs
            neighs = nn.kneighbors(vs_vectors)[1]
            del vs_vectors
            flat_neigh = neighs.flatten()
            indexes = np.repeat(np.arange(0, neighs.shape[0]), n_neighbors)
            pos_pairs = np.vstack([indexes, flat_neigh]).T
            # avoid identities
            pos_pairs = pos_pairs[pos_pairs[:, 0] != pos_pairs[:, 1]]
            pos_pairs = set([tuple(p) for p in pos_pairs])
            # get negative pairs
            idxs1 = np.repeat(np.arange(0, neighs.shape[0]), n_neighbors * 2)
            idxs2 = np.repeat(np.arange(0, neighs.shape[0]), n_neighbors * 2)
            np.random.shuffle(idxs2)
            neg_pairs = np.vstack([idxs1, idxs2]).T
            neg_pairs = neg_pairs[neg_pairs[:, 0] != neg_pairs[:, 1]]
            neg_pairs = set([tuple(p) for p in neg_pairs])
            neg_pairs = neg_pairs - pos_pairs
            neg_pairs = list(neg_pairs)[:len(pos_pairs)]

            # calculate distances of same pairs but in out signature
            my_vectors = self.sign.get_vectors(keys1)[1]
            # scipy computes the dot products in the input dtype: on integer
            # signatures (e.g. int8 sign0) they overflow and corrupt distances
            if np.issubdtype(my_vectors.dtype, np.integer):
                my_vectors = my_vectors.astype(np.float64)
            if metric == "cosine":
                from scipy.spatial.distance import cosine as metric
            if metric == "euclidean":
                from scipy.spatial.distance import euclidean as metric

            def _compute_dists(pairs):
                dists = list()
                for p1, p2 in pairs:
                    dists.append(metric(my_vectors[p1], my_vectors[p2]))
                dists = np.array(dists)
                # cosine distance in sparse matrices might be NaN or inf
                dists = dists[np.isfinite(dists)]
                return dists

            pos_dists = _compute_dists(pos_pairs)
            neg_dists = _compute_dists(neg_pairs)
            del my_vectors
            # correct negative/positive ratio
            if len(neg_dists) > len(pos_dists) * neg_pos_ratio:
                np.random.shuffle(neg_dists)
                neg_dists = neg_dists[:int(len(pos_dists) * neg_pos_ratio)]
            else:
                np.random.shuffle(pos_dists)
                pos_dists = pos_dists[:int(len(neg_dists) / neg_pos_ratio)]
            # final arrays for performance calculation
            y_t = np.array([1] * len(pos_dists) + [0] * len(neg_dists))
            y_p = np.hstack([pos_dists, neg_dists])
            # convert to similarity-respected order
            y_p = -np.abs(np.array(y_p))
            # roc calculation
            fpr, tpr, _ = roc_curve(y_t, y_p)
            # pr calculation
            precision, recall, _ = precision_recall_curve(y_t, y_p)
            # write results dictionary
            results = {
                "fpr": fpr,
                "tpr": tpr,
                "auc": auc(fpr, tpr),
                "precision": precision,
                "recall": recall,
                "average_precision_score": average_precision_score(y_t, y_p)
            }
        else:
            results = None
        if val_type == 'pr':
            plotter_function_arg = self.plotter.cross_pr
        else:
            plotter_function_arg = self.plotter.cross_roc
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=plotter_function_arg,
            kw_plotter={"sign": sign},
            **kwargs)

    # @safe_return(None)
    def projection(self, *args, keys=None, focus_keys=None, max_keys=10000,
                   perplexity=None, max_pca=100, redo=False, **kwargs):
        """TSNE projection of CC signatures.

        Args:
            keys (list): Keys to be projected. If None specified, keys are
                randomly sampled. (default=None)
            focus_keys (list): Keys to be highlighted in the projection.
                 (default=None).
            max_keys (int): Maximum number of keys to include in the
                projection. (default=10000)
        """
        self.__log.debug("Projection")
        fn = "projection"
        if self._todo(fn) or redo:
            from MulticoreTSNE import MulticoreTSNE as TSNE
            X, keys = self._get_signatures(keys=keys, max_keys=max_keys)
            self.__log.debug("Fit-transforming t-SNE")
            if X.shape[1] > max_pca:
                self.__log.debug("First doing a PCA")
                # check on max_pca value
                min_samples_features = min(X.shape[0], X.shape[1])
                if min_samples_features <= max_pca:
                    max_pca = min_samples_features
                X = PCA(n_components=max_pca).fit_transform(X)
            init = PCA(n_components=2).fit_transform(X)
            if perplexity is None:
                perp = int(np.sqrt(X.shape[0]))
                perp = np.max([5, perp])
                perp = np.min([50, perp])
            self.__log.debug("Chosen perplexity %d" % perp)
            tsne = TSNE(perplexity=perp, init=init, n_jobs=self.cpu)
            P_ = tsne.fit_transform(X)
            P = P_[:len(keys)]
            results = {
                "P": P,
                "keys": keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.projection,
            kw_plotter={'focus_keys': focus_keys},
            **kwargs)

    # @safe_return(None)
    def image(self, *args, keys=None, max_keys=100, shuffle=False, **kwargs):
        self.__log.debug("Image")
        fn = "image"
        if self._todo(fn):
            X, keys = self._get_signatures(keys=keys, max_keys=max_keys,
                                           shuffle=shuffle)
            results = {
                "X": X,
                "keys": keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.image,
            **kwargs)

    def _iqr(self, axis, keys, max_keys):
        X, keys = self._get_signatures(keys=keys, max_keys=max_keys)
        p25 = np.percentile(X, 25, axis=axis)
        p50 = np.percentile(X, 50, axis=axis)
        p75 = np.percentile(X, 75, axis=axis)
        results = {
            "p25": p25,
            "p50": p50,
            "p75": p75
        }
        return results

    # @safe_return(None)
    def features_iqr(self, *args, keys=None, max_keys=10000, **kwargs):
        self.__log.debug("Features IQR")
        fn = "features_iqr"
        if self._todo(fn):
            results = self._iqr(axis=0, keys=keys, max_keys=max_keys)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.features_iqr,
            **kwargs)

    # @safe_return(None)
    def keys_iqr(self, *args, keys=None, max_keys=1000, **kwargs):
        self.__log.debug("Keys IQR")
        fn = "keys_iqr"
        if self._todo(fn):
            results = self._iqr(axis=1, keys=keys, max_keys=max_keys)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.keys_iqr,
            **kwargs)

    def _bins(self, *args, axis, keys=None, max_keys=10000, n_bins=100,
              **kwargs):
        X, keys = self._get_signatures(keys=keys, max_keys=max_keys)
        bs = np.linspace(np.min(X), np.max(X), n_bins)
        H = np.apply_along_axis(lambda v: np.histogram(v, bins=bs)[0], axis, X)
        if axis == 1:
            H = H.T
        results = {
            "H": H,
            "bins": bs,
            "p50": np.median(X, axis=axis),
        }
        return results

    # @safe_return(None)
    def features_bins(self, *args, **kwargs):
        self.__log.debug("Features bins")
        fn = "features_bins"
        if self._todo(fn):
            results = self._bins(axis=0, **kwargs)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.features_bins,
            **kwargs)

    # @safe_return(None)
    def keys_bins(self, *args, **kwargs):
        self.__log.debug("Keys bins")
        fn = "keys_bins"
        if self._todo(fn):
            results = self._bins(axis=1, **kwargs)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.keys_bins,
            **kwargs)

    # @safe_return(None)
    def values(self, *args, max_values=10000, **kwargs):
        self.__log.debug("Values")
        fn = "values"
        if self._todo(fn):
            V = self.V.ravel()
            idxs = np.random.choice(len(V), min(max_values, len(V)),
                                    replace=False)
            V = V[idxs]
            kernel = gaussian_kde(V)
            positions = np.linspace(np.min(V), np.max(V), 1000)
            values = kernel(positions)
            results = {
                "x": positions,
                "y": values
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.values,
            **kwargs)

    def _projection_binned(self, P, scores, n_bins):
        # check nans
        mask = ~np.isnan(scores)
        P = P[mask, ]
        scores = scores[mask]
        self.__log.debug("Binning projection")
        bins_x = np.linspace(np.min(P[:, 0]), np.max(P[:, 0]) + 1e-6, n_bins)
        bins_y = np.linspace(np.min(P[:, 1]), np.max(P[:, 1]) + 1e-6, n_bins)
        H = np.zeros((len(bins_y), len(bins_x)))
        S = np.zeros(H.shape)
        for j in range(0, len(bins_x) - 1):
            min_x = bins_x[j]
            max_x = bins_x[j + 1]
            maskx = np.logical_and(P[:, 0] >= min_x, P[:, 0] < max_x)
            P_ = P[maskx]
            scores_ = scores[maskx]
            for i in range(0, len(bins_y) - 1):
                min_y = bins_y[i]
                max_y = bins_y[i + 1]
                masky = np.logical_and(P_[:, 1] >= min_y, P_[:, 1] < max_y)
                ss = scores_[masky]
                if len(ss) > 0:
                    H[i, j] = len(ss)
                    S[i, j] = np.mean(ss)
        bins_x = np.array([(bins_x[i] + bins_x[i + 1]) /
                           2 for i in range(0, len(bins_x) - 1)])
        bins_y = np.array([(bins_y[i] + bins_y[i + 1]) /
                           2 for i in range(0, len(bins_y) - 1)])
        results = {
            "H": H,
            "S": S,
            "bins_x": bins_x,
            "bins_y": bins_y,
            "scores": np.array([np.max(scores), np.min(scores)]),
            "lims": np.array([[np.min(P[:, 0]), np.min(P[:, 1])],
                              [np.max(P[:, 0]), np.max(P[:, 1])]])
        }
        return results

    # @safe_return(None)
    def confidences(self, *args, **kwargs):
        self.__log.debug("Confidences")
        fn = "confidences"
        if self._todo(fn):
            keys = self.keys
            mask = np.isin(
                list(self.sign.keys), list(keys), assume_unique=True)
            confidences = self.sign.get_h5_dataset('confidence')[mask]
            confidences = np.array(confidences)
            kernel = gaussian_kde(confidences)
            positions = np.linspace(
                np.min(confidences), np.max(confidences), 1000)
            values = kernel(positions)
            results = {
                "keys": self.keys,
                "confidences": confidences,
                "x": positions,
                "y": values
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.confidences,
            **kwargs)

    # @safe_return(None)
    def confidences_projection(self, *args, n_bins=20, **kwargs):
        self.__log.debug("Confidences projection")
        if self._todo("confidences", inner=True):
            raise Exception("confidences must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "confidences_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_inte = self._load_diagnosis_pickle("confidences.pkl")
            if np.any(results_proj["keys"] != results_inte["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_inte["confidences"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.confidences_projection,
            **kwargs)

    # @safe_return(None)
    def intensities(self, *args, **kwargs):
        self.__log.debug("Intensities")
        fn = "intensities"
        if self._todo(fn):
            V, keys = self._get_signatures(**kwargs)
            m = np.mean(V, axis=0)
            s = np.std(V, axis=0)
            # if std is zero we would get NaN, costant columns are irrelevant
            m = m[s != 0]
            V = V[:, s != 0]
            s = s[s != 0]
            norm = np.apply_along_axis(lambda v: np.abs((v - m) / s), 1, V)
            # for i in range(0, V.shape[0]):
            #     v = np.abs((V[i, :] - m) / s)
            #     intensities += [np.sum(v)]
            intensities = np.sum(norm, axis=1)
            kernel = gaussian_kde(intensities)
            positions = np.linspace(
                np.min(intensities), np.max(intensities), 1000)
            values = kernel(positions)
            results = {
                "keys": keys,
                "intensities": intensities,
                "x": positions,
                "y": values
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.intensities,
            **kwargs)

    # @safe_return(None)
    def intensities_projection(self, *args, n_bins=20, **kwargs):
        self.__log.debug("Intensities projection")
        if self._todo("intensities", inner=True):
            raise Exception("intensities must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "intensities_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_inte = self._load_diagnosis_pickle("intensities.pkl")
            if np.any(results_proj["keys"] != results_inte["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_inte["intensities"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.intensities_projection,
            **kwargs)

    def _latent(self, sign):
        fn = os.path.join(self.path, "latent-%d.pkl" % self.V.shape[0])
        if os.path.exists(fn):
            results = pickle.load(open(fn, "rb"))
        else:
            pca = PCA(n_components=0.9)
            V = sign.subsample(self.V.shape[0])[0]
            pca.fit(V)
            results = {"explained_variance": pca.explained_variance_ratio_}
        return results

    # @safe_return(None)
    def dimensions(self, *args, datasets=None, exemplary=True, molset="full", **kwargs):
        """Get dimensions of the signature and compare to other signatures."""
        self.__log.debug("Dimensions")
        fn = "dimensions"
        
        ref_cctype = 'sign1'
            
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                metadata = self.ref_cc.sign_metadata(
                    'dimensions', molset, ds, ref_cctype)
                if metadata is not None:
                    nr_keys, nr_feats = metadata
                    results[ds] = {
                        "keys": nr_keys,
                        "features": nr_feats
                    }
            results["MY"] = {
                "keys": self.sign.shape[0],
                "features": self.sign.shape[1]
            }
        else:
            results = None
        kw_plotter = {
            "exemplary": exemplary,
            "cctype": ref_cctype,
            "molset": molset}
        kw_plotter.update(kwargs)
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.dimensions,
            kw_plotter=kw_plotter,
            **kwargs)

    # @safe_return(None)
    def across_coverage(self, *args, datasets=None, exemplary=True, **kwargs):
        """Check coverage against a collection of other CC signatures.

        Args:
            datasets (list): List of datasets. If None, all available are used.
                (default=None)
            exemplary (bool): Whether to use only exemplary datasets
                (recommended). (default=True)
            cctype (str): CC signature type. (default=None)
            molset (str): Molecule set to use. Full is recommended.
                (default=None)
            kwargs (dict): params of hte cross_coverage method.
        """
        self.__log.debug("Across coverage")
        fn = "across_coverage"
        
        ref_cctype = 'sign1'
            
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                results[ds] = self.cross_coverage(
                    ds, ref_cctype=ref_cctype, save=False, redo=True,
                    plot=False)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.across_coverage,
            kw_plotter={
                "exemplary": exemplary,
                "cctype": ref_cctype},
            **kwargs)

    def _sample_accuracy_individual(self, sign, n_neighbors, min_shared,
                                    metric):
        # do nearest neighbors (start by the target signature)
        shared_keys = self._shared_keys(sign)
        keys, V1 = sign.get_vectors(shared_keys)
        if keys is None or len(keys) < min_shared:
            return None
        n_neighbors = np.min([V1.shape[0], n_neighbors])
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric,
                              n_jobs=self.cpu)
        nn.fit(V1)
        neighs1_ = nn.kneighbors(V1)[1][:, 1:]
        # do nearest neighbors for self
        mask = np.isin(list(self.keys), list(shared_keys), assume_unique=True)
        V0 = self.V[mask]
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric=metric,
                              n_jobs=self.cpu)
        nn.fit(V0)
        neighs0_ = nn.kneighbors(V0)[1][:, 1:]
        # reindex
        keys_dict = dict((k, i) for i, k in enumerate(self.keys))
        neighs0 = np.zeros(neighs0_.shape).astype(int)
        neighs1 = np.zeros(neighs1_.shape).astype(int)
        rows = []
        for i in range(0, neighs0_.shape[0]):
            rows += [keys_dict[keys[i]]]
            for j in range(0, neighs0_.shape[1]):
                neighs0[i, j] = keys_dict[keys[neighs0_[i, j]]]
                neighs1[i, j] = keys_dict[keys[neighs1_[i, j]]]
        return {"neighs0": neighs0, "neighs1": neighs1, "rows": rows}

    def _rbo(self, neighs0, neighs1, p):
        from chemicalchecker.tool.rbo.rbo import rbo
        scores = []
        for i in range(0, neighs0.shape[0]):
            scores += [rbo(neighs0[i, :], neighs1[i, :], p=p).ext]
        return np.array(scores)

    # @safe_return(None)
    def ranks_agreement(self, *args, datasets=None, exemplary=True,
                        ref_cctype="sign0", n_neighbors=100, min_shared=100,
                        metric="minkowski", p=0.9, **kwargs):
        """Sample-specific accuracy.

        Estimated as general agreement with the rest of the CC.
        """
        self.__log.debug("Sample-specific agreement to the rest of CC")
        fn = "ranks_agreement"

        if self.sign.shape[0] < min_shared:
            min_shared = 0.7 * self.sign.shape[0]
            # self.__log.debug("Not enough molecules in the dataset to generate ranks agreement: \n\
            #        dataset molecules {}, min_shared molecules {} ".format(self.sign.shape[0], min_shared))
            # return None

        if ref_cctype is None:
            ref_cctype = self.sign.cctype

        def q67(r):
            return np.percentile(r, 67)

        def stat(R, func):
            s = []
            for r in R:
                if len(r) == 0:
                    s += [np.nan]
                else:
                    s += [func(r)]
            return np.array(s)

        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results_datasets = {}
            for ds in datasets:
                sign = self.ref_cc.signature(ds, ref_cctype)
                dn = self._sample_accuracy_individual(
                    sign, n_neighbors, min_shared, metric)
                if dn is None:
                    continue
                sc = self._rbo(dn["neighs0"], dn["neighs1"], p=p)
                results_datasets[ds] = [r for r in zip(dn["rows"], sc)]
            d = {}
            for k in self.keys:
                d[k] = []
            for k, v in results_datasets.items():
                for r in v:
                    d[self.keys[r[0]]] += [(k, r[1])]
            R = []
            for k in self.keys:
                R += [d[k]]
            R_ = [[x[1] for x in r] for r in R]
            results = {
                "all": R,
                "max": stat(R_, np.max),
                "median": stat(R_, np.median),
                "mean": stat(R_, np.mean),
                "q67": stat(R_, q67),
                "size": np.array([len(r) for r in R_]),
                "keys": self.keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.ranks_agreement,
            kw_plotter={
                "exemplary": exemplary,
                "cctype": ref_cctype},
            **kwargs)

    # @safe_return(None)
    def ranks_agreement_projection(self, *args, n_bins=20, **kwargs):
        self.__log.debug("Ranks agreement projection")
        if self._todo("ranks_agreement", inner=True):
            raise Exception("ranks_agreement must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "ranks_agreement_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_rnks = self._load_diagnosis_pickle("ranks_agreement.pkl")
            if np.any(results_proj["keys"] != results_rnks["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_rnks["mean"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.ranks_agreement_projection,
            **kwargs)

    # @safe_return(None)
    def global_ranks_agreement(self, *args, n_neighbors=100, min_shared=100,
                               metric="minkowski", p=0.9, ref_cctype=None,
                               **kwargs):
        """Sample-specific global accuracy.

        Estimated as general agreement with the rest of the CC, based on a
        Z-global ranking.
        """
        self.__log.debug(
            "Sample-specific agreement to the rest of CC,"
            " based on a Z-global ranking")
        fn = "global_ranks_agreement"

        # to take into consideration the case of very small datasets
        # with less than min_shared molecules
        if self.sign.shape[0] < min_shared:
            # TODO: is this plot significant anyways?
            min_shared = 0.7 * self.sign.shape[0]
            # self.__log.debug("Not enough molecules in the dataset to generate global ranks agreement: \n\
            #        dataset molecules {}, min_shared molecules {} ".format(self.sign.shape[0], min_shared))
            # return None

        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        exemplary_datasets = self._select_datasets(None, True)
        datasets = kwargs.get("datasets", exemplary_datasets)

        def q67(r):
            return np.percentile(r, 67)

        def stat(R, func):
            s = []
            for r in R:
                if len(r) == 0:
                    s += [np.nan]
                else:
                    s += [func(r)]
            return np.array(s)

        if self._todo(fn):
            results_datasets = {}
            for ds in datasets:
                sign = self.ref_cc.signature(ds, ref_cctype)
                dn = self._sample_accuracy_individual(
                    sign, n_neighbors, min_shared, metric)
                if dn is None:
                    continue
                sc = self._rbo(dn["neighs0"], dn["neighs1"], p=p)
                results_datasets[ds] = [r for r in zip(dn["rows"], sc)]
            d = {}
            for k in self.keys:
                d[k] = []
            for k, v in results_datasets.items():
                for r in v:
                    d[self.keys[r[0]]] += [(k, r[1])]
            R = []
            for k in self.keys:
                R += [d[k]]
            R_ = [[x[1] for x in r] for r in R]
            results = {
                "all": R,
                "max": stat(R_, np.max),
                "median": stat(R_, np.median),
                "mean": stat(R_, np.mean),
                "q67": stat(R_, q67),
                "size": np.array([len(r) for r in R_]),
                "keys": self.keys
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.global_ranks_agreement,
            **kwargs)

    # @safe_return(None)
    def global_ranks_agreement_projection(self, *args, n_bins=20, **kwargs):
        self.__log.debug("Global ranks agreement projection")
        if self._todo("global_ranks_agreement", inner=True):
            raise Exception("global_ranks_agreement must be done first")
        if self._todo("projection", inner=True):
            raise Exception("projection must be done first")
        fn = "global_ranks_agreement_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_rnks = self._load_diagnosis_pickle(
                "global_ranks_agreement.pkl")
            if np.any(results_proj["keys"] != results_rnks["keys"]):
                raise Exception("keys do not coincide...")
            results = self._projection_binned(
                results_proj["P"], results_rnks["mean"], n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.global_ranks_agreement_projection,
            **kwargs)

    # @safe_return(None)
    def across_roc(self, *args, datasets=None, exemplary=True, ref_cctype=None,
                   redo=False, include_datasets=None, **kwargs):
        """Check coverage against a collection of other CC signatures.

        Args:
            datasets (list): List of datasets. If None, all available are used.
                (default=None).
            exemplary (bool): Whether to use only exemplary datasets
                (recommended). (default=True)
            ref_cctype (str): CC signature type. (default='sign0')
            redo (bool): redo the plot
            include_datasets (list): specific datasets to add when exemplary 
                                     is set to True (default=None)
            kwargs (dict): Parameters of the cross_roc method.
        """
        self.__log.debug("Across ROC")
        fn = "across_roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if self._todo(fn) or redo:
            print('here')
            datasets = self._select_datasets(datasets, exemplary)
            results = {}
            for ds in datasets:
                sign = self.ref_cc.signature(ds, ref_cctype)
                print( 'across', sign)
                results[ds] = self.cross_roc(sign, save=False, redo=True,
                                             plot=False)
            if include_datasets and exemplary:
                for incl_ds in include_datasets:
                    sign = self.ref_cc.signature(incl_ds, ref_cctype)
                    results[incl_ds] = self.cross_roc(sign, save=False, redo=True,
                                                      plot=False)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.across_roc,
            kw_plotter={
                "exemplary": exemplary,
                "cctype": ref_cctype},
            **kwargs)

    # @safe_return(None)
    def neigh_roc(self, ds, *args, ref_cctype=None,
                  n_neighbors=[1, 5, 10, 50, 100], **kwargs):
        """Check ROC against another signature at different NN levels.

        Args:
            ds: Dataset aginst which to run ROC analysis.
            ref_cctype (str): CC signature type.
            neighbors (list): list of top NN for which we want to compute ROC.
            molset (str): Molecule set to use. Full is recommended.
                (default='full')
            kwargs (dict): Parameters of hte cross_coverage method.
        """
        self.__log.debug("Multiple Neighbors ROC")
        fn = "neigh_roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if self._todo(fn):
            results = {}
            sign = self.ref_cc.signature(ds, ref_cctype)
            for nn in n_neighbors:
                results[nn] = self.cross_roc(
                    sign, save=False, redo=True, n_neighbors=nn, plot=False)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.neigh_roc,
            kw_plotter={
                "ds": ds},
            **kwargs)

    # @safe_return(None)
    def atc_roc(self, *args, ref_cctype=None, redo=False, **kwargs):
        self.__log.debug("ATC ROC")
        fn = "atc_roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        ds = "E1.001"
        if self._todo(fn) or redo:
            sign = self.ref_cc.signature(ds, ref_cctype)
            results = self.cross_roc(sign, redo=True, save=False, plot=False)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.atc_roc,
            **kwargs)

    # @safe_return(None)
    def moa_roc(self, *args, ref_cctype=None, redo=False, **kwargs):
        self.__log.debug("MoA ROC")
        fn = "moa_roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        ds = "B1.001"
        if self._todo(fn) or redo:
            sign = self.ref_cc.signature(ds, ref_cctype)
            results = self.cross_roc(sign, redo=True, save=False, plot=False)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.moa_roc,
            **kwargs)

    # @safe_return(None)
    def roc(self, ds, *args, ref_cctype=None, redo=False, **kwargs):
        self.__log.debug("ROC")
        fn = "roc"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if self._todo(fn) or redo:
            sign = self.ref_cc.signature(ds, ref_cctype)
            results = self.cross_roc(
                sign, redo=True, save=False, plot=False, val_type='roc')
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.roc,
            kw_plotter={"ds": ds},
            **kwargs)

    # @safe_return(None)
    def pr(self, ds, *args, ref_cctype=None, redo=False, **kwargs):
        self.__log.debug("PrecisionRecall")
        fn = "pr"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if self._todo(fn) or redo:
            sign = self.ref_cc.signature(ds, ref_cctype)
            results = self.cross_roc(
                sign, redo=True, save=False, plot=False, val_type='pr')
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.pr,
            kw_plotter={"ds": ds},
            **kwargs)

    # @safe_return(None)
    def redundancy(self, *args, **kwargs):
        self.__log.debug("Redundancy")
        fn = "redundancy"
        if self._todo(fn):
            from chemicalchecker.util.remove_near_duplicates import RNDuplicates
            self.__log.debug("Removing near duplicates")
            rnd = RNDuplicates(cpu=self.cpu)
            mappings = rnd.remove(self.sign.data_path,
                                  save_dest=None, just_mappings=True)
            mappings = np.array(sorted(mappings.items()))
            mps = []
            for i in range(0, mappings.shape[0]):
                mps += [(mappings[i, 0], mappings[i, 1])]
            mps = np.array(mps)
            n_full = mps.shape[0]
            n_ref = len(set(mps[:, 1]))
            red_counts = collections.defaultdict(int)
            for i in range(0, mps.shape[0]):
                red_counts[mps[i, 1]] += 1
            red_counts = [(k, v) for k, v in sorted(
                red_counts.items(), key=lambda item: -item[1])]
            results = {
                "n_full": n_full,
                "n_ref": n_ref,
                "counts": red_counts
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.redundancy,
            **kwargs)

    def _cluster(self, expected_coverage, n_neighbors, min_samples,
                 top_clusters):
        self.__log.debug("Clustering")
        P = self._load_diagnosis_pickle("projection.pkl")["P"]
        min_samples = int(
            np.max([np.min([P.shape[0] * 0.01, min_samples]), 5]))
        self.__log.debug("Estimating epsilon")
        if n_neighbors is None:
            n_neighbors = [1, 2, 3, 5, 10]
        else:
            n_neighbors = [n_neighbors]

        def do_clustering(n_neigh):
            nn = NearestNeighbors(n_neighbors=n_neigh + 1, n_jobs=self.cpu)
            nn.fit(P)
            dists = nn.kneighbors(P)[0][:, n_neigh]
            # FIXME: bins='auto' is preferable, but might trigger memory
            # errors e.g. C3 sign1
            h = np.histogram(dists, bins=100)
            y = np.cumsum(h[0]) / np.sum(h[0])
            x = h[1][1:]
            eps = x[np.where(y > expected_coverage)[0][0]]
            self.__log.debug("Running DBSCAN")
            cl = DBSCAN(eps=eps, min_samples=min_samples,
                        n_jobs=self.cpu).fit(P)
            labels = cl.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)
            lab_counts = collections.defaultdict(int)
            for lab in labels:
                lab_counts[lab] += 1
            lab_counts = [(k, v) for k, v in sorted(
                lab_counts.items(), key=lambda item: -item[1])]
            return (labels, lab_counts, eps, n_clusters_, n_noise_)
            
        best_score = 0
        best_n_neigh = None
        for n_neigh in n_neighbors:
            labels, lab_counts, eps, n_clusters_, n_noise_ = do_clustering(
                n_neigh)
            lc = list([x[1] for x in lab_counts[:top_clusters]])
            if len(lc) < top_clusters:
                lc += [0] * (top_clusters - len(lc))
            score = np.median(lc)
            if best_n_neigh is None:
                best_n_neigh = n_neigh
                best_score = score
            if best_score < score:
                best_n_neigh = n_neigh
                best_score = score
        labels, lab_counts, eps, n_clusters_, n_noise_ = do_clustering(
            best_n_neigh)
        results = {
            "P": P,
            "min_samples": min_samples,
            "labels": labels,
            "lab_counts": lab_counts,
            "epsilon": eps,
            "n_clusters": n_clusters_,
            "n_noise": n_noise_
        }
        return results

    # @safe_return(None)
    def cluster_sizes(self, *args, expected_coverage=0.95, n_neighbors=None,
                      min_samples=10, top_clusters=20, **kwargs):
        if self._todo("projection", inner=True):
            raise Exception("projection needs to be done first")
        self.__log.debug("Cluster sizes")
        fn = "clusters"
        if self._todo(fn):
            results = self._cluster(
                expected_coverage, n_neighbors, min_samples, top_clusters)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.cluster_sizes,
            **kwargs)

    # @safe_return(None)
    def clusters_projection(self, *args, **kwargs):
        self.__log.debug("Projection clusters")
        if self._todo("projection", inner=True):
            raise Exception("projection needs to be done first")
        fn = "clusters"
        if self._todo(fn, inner=True):
            raise Exception("clusters needs to be done first")
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.clusters_projection,
            **kwargs)

    # @safe_return(None)
    def key_coverage(self, *args, datasets=None, exemplary=True, molset='full', **kwargs):
        self.__log.debug("Key coverages")
        
        ref_cctype = 'sign1'
            
        fn = "key_coverage"
        if ref_cctype is None:
            ref_cctype = self.ref_cctype
        if self._todo(fn):
            datasets = self._select_datasets(datasets, exemplary)
            results_shared = {}
            for ds in datasets:
                metadata = self.ref_cc.sign_metadata(
                    'keys', molset, ds, ref_cctype)
                if metadata is not None:
                    sign_keys = metadata
                    results_shared[ds] = sorted(
                        list(set(self.keys) & set(sign_keys)))
            results_counts = {}
            for k in self.keys:
                results_counts[k] = 0
            for ds, v in results_shared.items():
                for k in v:
                    results_counts[k] += 1
            results = {
                "shared": results_shared,
                "counts": results_counts,
                "key_counts": np.array([results_counts[k] for k in self.keys])
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.key_coverage,
            kw_plotter={
                "exemplary": exemplary
            },
            **kwargs)

    # @safe_return(None)
    def key_coverage_projection(self, *args, n_bins=20, **kwargs):
        if self._todo("key_coverage", inner=True):
            raise Exception("key_coverage needs to be run first")
        if self._todo("projection", inner=True):
            raise Exception("projection needs to be done first")
        fn = "key_coverage_projection"
        if self._todo(fn):
            results_proj = self._load_diagnosis_pickle("projection.pkl")
            results_cov = self._load_diagnosis_pickle("key_coverage.pkl")
            results = self._projection_binned(results_proj["P"], np.array(
                results_cov["key_counts"]), n_bins=n_bins)
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.key_coverage_projection,
            **kwargs)

    # @safe_return(None)
    def orthogonality(self, *args, max_features=1000, **kwargs):
        fn = "orthogonality"
        if self._todo(fn):
            V, keys = self._get_signatures(max_features=max_features, **kwargs)
            V = normalize(V, norm="l2", axis=0)
            dots = []
            for i in range(0, V.shape[1] - 1):
                for j in range(i + 1, V.shape[1]):
                    dots += [np.dot(V[:, i], V[:, j].T)]
            results = {
                "dots": np.array(dots)
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.orthogonality,
            **kwargs)

    # @safe_return(None)
    def outliers(self, *args, n_estimators=1000, **kwargs):
        """Computes anomaly score of the input samples.

        The lower, the more abnormal. Negative scores represent outliers,
        positive scores represent inliers.
        """
        fn = "outliers"
        if self._todo(fn):
            max_features = int(np.sqrt(self.V.shape[1]) + 1)
            max_samples = min(1000, int(self.V.shape[0] / 2 + 1))
            mod = IsolationForest(n_estimators=n_estimators, contamination=0.1,
                                  max_samples=max_samples,
                                  max_features=max_features, n_jobs=self.cpu)
            pred = mod.fit_predict(self.V)
            scores = mod.decision_function(self.V)
            results = {
                "scores": scores,
                "pred": pred
            }
        else:
            results = None
        return self._returner(
            results=results,
            fn=fn,
            plotter_function=self.plotter.outliers,
            **kwargs)

    def available(self):
        return self.plotter.available()

    def canvas_small(self, title=None, skip_plots=[]):
        shared_kw = dict(save=True, plot=False, ref_cctype=self.ref_cctype )
        if "cosine_distances" not in skip_plots:
            self.cosine_distances(**shared_kw)
        if "euclidean_distances" not in skip_plots:
            self.euclidean_distances(**shared_kw)
        if "projection" not in skip_plots:
            self.projection(**shared_kw)
        if "image" not in skip_plots:
            self.image(**shared_kw)
        if "features_bins" not in skip_plots:
            self.features_bins(**shared_kw)
        if "keys_bins" not in skip_plots:
            self.keys_bins(**shared_kw)
        if "values" not in skip_plots:
            self.values(**shared_kw)
        if "redundancy" not in skip_plots:
            self.redundancy(**shared_kw)

        fig = self.plotter.canvas(
            size="small", title=title, skip_plots=skip_plots)
        return fig

    def canvas_medium(self, title=None, skip_plots=[]):
        available_sign = self.ref_cc.report_available(
            molset="full", signature=self.ref_cctype)
        if len(available_sign["full"]) < 25:
            self.__log.warning(
                "[ERROR] Reference CC '%s' does not have enough '%s'" %
                (self.ref_cc.name, self.ref_cctype))
            raise Exception("You do not have enough signatures for this cc type - Get them on https://chemicalchecker.com/downloads/root")
            
        shared_kw = dict(save=True, plot=False, ref_cctype=self.ref_cctype )

        if "cosine_distances" not in skip_plots:
            self.cosine_distances(**shared_kw)
        if "euclidean_distances" not in skip_plots:
            self.euclidean_distances(**shared_kw)
        if "projection" not in skip_plots:
            self.projection(**shared_kw)
        if "image" not in skip_plots:
            self.image(**shared_kw)
        if "features_bins" not in skip_plots:
            self.features_bins(**shared_kw)
        if "keys_bins" not in skip_plots:
            self.keys_bins(**shared_kw)
        if "values" not in skip_plots:
            self.values(**shared_kw)
        if "redundancy" not in skip_plots:
            self.redundancy(**shared_kw)
        if "intensities" and "intensities_projection" not in skip_plots:
            self.intensities(**shared_kw)
            if "intensities_projection" not in skip_plots:
                self.intensities_projection(**shared_kw)
        if self.sign.cctype == 'sign3':
            if "confidences" not in skip_plots:
                self.confidences(**shared_kw)
                if "confidences_projection" not in skip_plots:
                    self.confidences_projection(**shared_kw)
        if "cluster_sizes" not in skip_plots:
            self.cluster_sizes(**shared_kw)
            if "clusters_projection" not in skip_plots:
                self.clusters_projection(**shared_kw)
        if "outliers" not in skip_plots:
            self.outliers(**shared_kw)

        # these plots requires CC wide metadata
        if "dimensions" not in skip_plots:
            self.dimensions(**shared_kw)
        if "key_coverage" not in skip_plots:
            self.key_coverage(**shared_kw)
            if "key_coverage_projection" not in skip_plots:
                self.key_coverage_projection(**shared_kw)
        if "across_coverage" not in skip_plots:
            self.across_coverage(**shared_kw)
        # these plots requires CC wide signatures
        if "atc_roc" not in skip_plots:
            self.atc_roc(**shared_kw)
        if "moa_roc" not in skip_plots:
            self.moa_roc(**shared_kw)
        if "across_roc" not in skip_plots:
            self.across_roc(**shared_kw)
        if self.global_ranks_agreement(**shared_kw) is None:
            self.__log.debug(
                "Skipping plots Global Ranks Agreement and Global Ranks Agreement Projection")
            skip_plots.extend("global_ranks_agreement", 
                              "global_ranks_agreement_projection")
        else:
            self.global_ranks_agreement_projection(**shared_kw)

        fig = self.plotter.canvas(
            size="medium", title=title, skip_plots=skip_plots)
        return fig

    def custom_comparative_vertical(self, title=None, skip_plots=[]):
        shared_kw = dict(save=True, plot=False)

        self.projection(**shared_kw)
        self.intensities(**shared_kw)
        if self.sign.cctype == 'sign3':
            self.confidences(**shared_kw)
        self.clusters_projection(**shared_kw)
        self.atc_roc(**shared_kw)
        self.moa_roc(**shared_kw)
        self.across_roc(**shared_kw)

        fig = self.plotter.canvas(
            size="compare_v", title=title, skip_plots=skip_plots)
        return fig

    def canvas_large(self, title=None, skip_plots=[]):
        pass

    def canvas(self, size="medium", title=None, savefig=False, dest_dir=None,
               savefig_kwargs={'facecolor': 'white'}, skip_plots=[]):
        self.__log.debug("Computing or retrieving data for canvas %s." % size)
        self.__log.debug("Skipping the following plots: %s" % skip_plots)
        if size == "small":
            fig = self.canvas_small(title=title, skip_plots=skip_plots)
        elif size == "medium":
            fig = self.canvas_medium(title=title, skip_plots=skip_plots)
        elif size == "large":
            fig = self.canvas_large(title=title, skip_plots=skip_plots)
        elif size == "compare_v":
            fig = self.custom_comparative_vertical(
                title=title, skip_plots=skip_plots)
        else:
            return None
        if savefig:
            fn = "_".join([self.sign.dataset, self.sign.cctype,
                           self.name, size]) + '.png'
            if dest_dir is None:
                fn_dest = os.path.join(self.path, fn)
            else:
                fn_dest = os.path.join(dest_dir, fn)
            self.__log.debug("Saving plot to: %s" % fn_dest)
            fig.savefig(fn_dest, **savefig_kwargs)
        return fig
