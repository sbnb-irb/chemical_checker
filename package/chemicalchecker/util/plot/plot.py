"""Plot information on single Chemical Checker dataset."""
from __future__ import division

import os
import sys
import math
import h5py
import random
import inspect
import functools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, fisher_exact, gaussian_kde
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics import roc_curve, roc_auc_score, r2_score
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.gridspec as gridspec

from chemicalchecker.util import logged
from chemicalchecker.util import Config

random.seed(42)
np.random.seed(42)


def skip_on_exception(function):
    """Assist in skipping failing plots gracefully."""
    @logged
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as e:
            # log the exception
            err = "There was an exception in  "
            err += function.__name__ + ": "
            err += str(e)
            wrapper._log.error(err)
            return None
    return wrapper


def _rgb2hex(r, g, b):
    return '#%02x%02x%02x' % (r, g, b)


def coord_color(coordinate):
    if coordinate[0] == 'A':
        return _rgb2hex(250, 100, 80)
    if coordinate[0] == 'B':
        return _rgb2hex(200, 100, 225)
    if coordinate[0] == 'C':
        return _rgb2hex(80, 120, 220)
    if coordinate[0] == 'D':
        return _rgb2hex(120, 180, 60)
    if coordinate[0] == 'E':
        return _rgb2hex(250, 150, 50)

    return _rgb2hex(250, 100, 80)


def lighten_color(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


@logged
class Plot():
    """Plot class.

    Produce different kind of plots.
    """

    def __init__(self, dataset, plot_path, validation_path=None, svg=True):
        """Initialize a Plot instance.

        Produce all kind of plots and data associated

        Args:
            dataset(str): A Dataset object or the dataset code.
            plot_path(str): Final destination for the new plot and other stuff.
            validation_path(str): Directory where validation stats are found.
        """
        if not os.path.isdir(plot_path):
            raise Exception("Folder to save plots does not exist")
        if hasattr(dataset, 'code'):
            dataset_code = dataset.dataset_code
        else:
            dataset_code = dataset
        self.__log.debug('Plots for %s saved to %s',
                         dataset_code, plot_path)
        self.plot_path = plot_path
        if validation_path is None:
            try:
                self.validation_path = Config().PATH.validation_path
            except:
                self.validation_path = ""
        else:
            self.validation_path = validation_path
        self.dataset_code = dataset_code
        self.color = self._coord_color(dataset_code)

    def _elbow(self, curve):
        from numpy import matlib
        nPoints = len(curve)
        allCoord = np.vstack((range(nPoints), curve)).T
        np.array([range(nPoints), curve])
        firstPoint = allCoord[0]
        lineVec = allCoord[-1] - allCoord[0]
        lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
        vecFromFirst = allCoord - firstPoint
        scalarProduct = np.sum(
            vecFromFirst * matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
        vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
        vecToLine = vecFromFirst - vecFromFirstParallel
        distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
        idxOfBestPoint = np.argmax(distToLine)
        return idxOfBestPoint

    def _rgb2hex(self, r, g, b):
        return '#%02x%02x%02x' % (r, g, b)

    def _coord_color(self, coordinate):
        A = self._rgb2hex(250, 100, 80)
        B = self._rgb2hex(200, 100, 225)
        C = self._rgb2hex(80, 120, 220)
        D = self._rgb2hex(120, 180, 60)
        E = self._rgb2hex(250, 150, 50)
        if coordinate[0] == 'A':
            return A
        if coordinate[0] == 'B':
            return B
        if coordinate[0] == 'C':
            return C
        if coordinate[0] == 'D':
            return D
        if coordinate[0] == 'E':
            return E

        return A

    @staticmethod
    def cc_colors(coord, lighness=0):
        colors = {
            'A': ['#EA5A49', '#EE7B6D', '#F7BDB6'],
            'B': ['#B16BA8', '#C189B9', '#D0A6CB'],
            'C': ['#5A72B5', '#7B8EC4', '#9CAAD3'],
            'D': ['#7CAF2A', '#96BF55', '#B0CF7F'],
            'E': ['#F39426', '#F5A951', '#F8BF7D'],
            'Z': ['#000000', '#666666', '#999999']}
        return colors[coord[:1]][lighness]

    def clustering_plot(self, Nc, A, B, C):

        sns.set_style("white")

        plt.figure(figsize=(4, 4), dpi=600)

        fig = plt.subplot(111)
        plt.plot(Nc, A, color=self.color, lw=1, linestyle=':')
        plt.plot(Nc, B, color=self.color, lw=1, linestyle='--')
        plt.plot(Nc, C, color=self.color, lw=2, linestyle='-')

        kidx = np.argmax(C)
        k = Nc[kidx]
        plt.plot([k, k], [0, 1], color=self.color, lw=1, linestyle='-')

        plt.ylim(0, 1)
        plt.xlim(np.min(Nc), np.max(Nc))
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Cluster measures")

        plt.title("k: %d clusters" % k)

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/kmeans_kselect.png" % self.plot_path)

        with open("%s/kmeans_kselect.tsv" % self.plot_path, "w") as f:
            for i in range(len(Nc)):
                f.write("%d\t%f\t%f\t%f\n" % (Nc[i], A[i], B[i], C[i]))

        return k

    def variance_plot(self, exp_var_ratios, variance_cutoff=0.9):

        # Set the variance-explained cutoff.

        sns.set_style("white")

        cumsum = np.cumsum(exp_var_ratios)

        plt.figure(figsize=(4, 4), dpi=600)

        fig = plt.subplot(111)
        x = [i for i in range(len(cumsum) + 1)]
        y = [0] + list(cumsum)
        plt.plot(x, y, color=self.color, lw=2)

        # Varcut
        varcut = variance_cutoff
        for i in range(len(cumsum)):
            if cumsum[i] > varcut:
                break
        i90 = i
        plt.plot([0., i90 + 1], [varcut, varcut],
                 color=self.color, linestyle="-")
        plt.plot([i90 + 1, i90 + 1], [0, varcut],
                 color=self.color, linestyle="-")
        plt.scatter([i90 + 1], [varcut], color="white",
                    edgecolor=self.color, lw=1.5, zorder=3, s=50)

        # Elbow point
        curve = cumsum[:i90 + 1]
        ielb = self._elbow(curve)
        elb = cumsum[ielb]
        plt.plot([0., ielb + 1], [elb, elb], color=self.color, linestyle="--")
        plt.plot([ielb + 1, ielb + 1], [0, elb],
                 color=self.color, linestyle="--")
        plt.scatter([ielb + 1], [elb], color="white",
                    edgecolor=self.color, lw=1.5, zorder=3, s=50)

        plt.grid(linestyle="-.", color=self.color, lw=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, len(cumsum))
        plt.xlabel("Latent variables")
        plt.ylabel("Proportion of variance explained")

        plt.title("%.1f: %d, elbow: %d" % (varcut, i90 + 1, ielb + 1))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/variance_explained.png" % self.plot_path)

        with open("%s/variance_explained.tsv" % self.plot_path, "w") as f:
            for i in range(len(x)):
                f.write("%f\t%f\n" % (x[i], y[i]))

        return i90, ielb

    def get_same_different(self, sign, prefix, mappings=None, max_diff=5):
        """Return pairs of SAME and DIFFERENT molecules from validation set."""
        # read validation sets
        # entry are pairs of inchikeys that can be same (1) or not (0)
        filename = os.path.join(self.validation_path,
                                "%s_validation.tsv" % prefix)
        same = set()
        different = set()
        validation_inks = set()
        with open(filename, 'r') as fh:
            for line in fh:
                line = line.rstrip("\n").split("\t")
                ink0 = line[0]
                ink1 = line[1]
                validation_inks.add(ink0)
                validation_inks.add(ink1)
                if int(line[2]) == 1:
                    same.add((ink0, ink1))
                    continue
                different.add((ink0, ink1))
        # log validations set composition
        self.__log.info("%s concerns %s molecules", prefix.upper(),
                        len(validation_inks))
        self.__log.info("%s pairs with SAME %s", len(same),
                        prefix.upper())
        self.__log.info("%s pairs with DIFFERENT %s", len(different),
                        prefix.upper())
        # if there's no mapping it's faster
        if mappings is None:
            # find shared molecules between signature and validation set
            shared_inks = validation_inks & sign.unique_keys
            self.__log.info("shares with signature %s molecules",
                            len(shared_inks))
            if len(shared_inks) == 0:
                return [], [], [], []
            # get signature for all shared molecule
            all_signs_dict = dict(zip(*sign.get_vectors(shared_inks)))
            # only consider pairs for available molecules
            same_shared = list()
            for ink0, ink1 in same:
                if ink0 in shared_inks and ink1 in shared_inks:
                    same_shared.append((ink0, ink1))
            different_shared = list()
            for ink0, ink1 in different:
                if ink0 in shared_inks and ink1 in shared_inks:
                    different_shared.append((ink0, ink1))
        else:
            # find shared molecules between signature and validation set
            shared_inks = validation_inks & set(mappings.keys())
            self.__log.info("shares with signature %s molecules (mappings)",
                            len(shared_inks))
            if len(shared_inks) == 0:
                return [], [], [], []
            # no shortcut, let's go one by one
            all_signs_dict = dict()
            # only consider pairs for available molecules
            same_shared = list()
            for ink0, ink1 in same:
                if ink0 in shared_inks and ink1 in shared_inks:
                    same_shared.append((ink0, ink1))
                    if ink0 not in all_signs_dict:
                        all_signs_dict[ink0] = sign[mappings[ink0]]
                    if ink1 not in all_signs_dict:
                        all_signs_dict[ink1] = sign[mappings[ink1]]
            different_shared = list()
            for ink0, ink1 in different:
                if ink0 in shared_inks and ink1 in shared_inks:
                    different_shared.append((ink0, ink1))
                    if ink0 not in all_signs_dict:
                        all_signs_dict[ink0] = sign[mappings[ink0]]
                    if ink1 not in all_signs_dict:
                        all_signs_dict[ink1] = sign[mappings[ink1]]
        self.__log.info("%s shared pairs with SAME %s", len(same_shared),
                        prefix.upper())
        self.__log.info("%s shared pairs with DIFFERENT %s",
                        len(different_shared), prefix.upper())
        # cap set of different molecules
        if len(different_shared) > max_diff * len(same_shared):
            self.__log.info("limiting DIFFERENT pairs at %s times SHARED",
                            max_diff)
            limit = max_diff * len(same_shared)
            different_shared = set(list(different_shared)[:int(limit)])
        frac_shared = 100 * len(shared_inks) / float(len(validation_inks))
        return same_shared, different_shared, all_signs_dict, frac_shared

    def label_validation(self, inchikey_lab, label_type, prefix="moa", inchikey_mappings=None):

        S, D, d, frac = self.get_same_different(
            inchikey_lab, prefix, inchikey_mappings)

        yy, yn, ny, nn = 0, 0, 0, 0

        for k in S:
            if d[k[0]] == d[k[1]]:
                yy += 1
            else:
                yn += 1
        for k in D:
            if d[k[0]] == d[k[1]]:
                ny += 1
            else:
                nn += 1

        M = np.array([[yy, yn], [ny, nn]])

        odds, pval = fisher_exact(M, alternative="greater")

        sns.set_style("white")
        plt.figure(figsize=(4, 4), dpi=600)
        fig = plt.subplot(111)
        plt.bar([1, 2], [M[0, 0] / (M[1, 0] + M[0, 0]) * 100, M[0, 1] /
                         (M[1, 1] + M[0, 1]) * 100], color=[self.color, "white"], edgecolor=self.color, lw=2)
        plt.xticks([1, 2], ["Same", "Different"])
        for h in np.arange(10, 100, 10):
            plt.plot([-1, 3], [h, h], linestyle='--', color=self.color, lw=0.3)
        plt.ylim((0, 100))
        plt.xlim((0.5, 2.5))
        plt.ylabel("% in same cluster")

        plt.title("Odds: %.2f, P-val: %.2g" % (odds, pval))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.tight_layout()
        plt.savefig("%s/%s_%s_ft_validation.png" %
                    (self.plot_path, prefix, label_type))

        with open("%s/%s_%s_ft_validation.tsv" % (self.plot_path, prefix, label_type), "w") as f:
            S = "%s\t%d\n" % ("yy", yy)
            S += "%s\t%d\n" % ("yn", yn)
            S += "%s\t%d\n" % ("ny", ny)
            S += "%s\t%d\n" % ("nn", nn)
            S += "odds\t%.2f\n" % (odds)
            S += "pval\t%.2g\n" % (pval)
            f.write(S)

        return odds, pval

    def vector_validation(self, sign, cctype, prefix="moa", distance="cosine",
                          mappings=None):

        self.__log.info("%s Validation" % prefix.upper())
        S, D, d, frac = self.get_same_different(
            sign, prefix, mappings)

        if len(S) == 0 or len(D) == 0:
            self.__log.warn("Not enough pairs to validate...")
            return (-999,-999), -999, -999

        if distance == "euclidean":
            distance_metric = euclidean
        elif distance == "cosine":
            distance_metric = cosine
        else:
            raise Exception("Unrecognized distance %s" % distance)

        # scipy computes the dot products in the input dtype: on integer
        # signatures (e.g. int8 sign0) they overflow and corrupt distances
        d = {k: v.astype(np.float64) if np.issubdtype(v.dtype, np.integer)
             else v for k, v in d.items()}

        S = np.array(sorted([distance_metric(d[k[0]], d[k[1]]) for k in S]))
        D = np.array(sorted([distance_metric(d[k[0]], d[k[1]]) for k in D]))

        # This exception prevents this eeror in python3 on our pytest:
        # ValueError: zero-size array to reduction operation maximum which has
        # no identity
        try:
            ks = ks_2samp(S, D)
        except ValueError:
            ks = [0.0, 0.0]
            pass

        N = len(d)

        d = None

        # Distance plot

        sns.set_style("white")

        plt.figure(figsize=(4, 4), dpi=600)
        fig = plt.subplot(111)

        cS = np.cumsum(S)
        cS = cS / np.max(cS)

        cD = np.cumsum(D)
        cD = cD / np.max(cD)

        plt.plot(D, cD, color=self.color, linestyle="--")
        plt.plot(S, cS, color=self.color, linestyle="-", lw=2)

        plt.grid(linestyle="-.", color=self.color, lw=0.3)

        plt.ylim(0, 1)
        # Nico 19-07-2020: exclude Inf or NaN, in max and min, otherwise Matplotlib error.
        def excludeNanInf(nparr):
            return nparr[np.isfinite(nparr)]

        S= excludeNanInf(S)
        D= excludeNanInf(D)
        #---
        plt.xlim(np.min([np.min(S), np.min(D)]),
                 np.max([np.max(S), np.max(D)]))


        plt.xlabel("Distance")
        plt.ylabel("Cumulative proportion")

        plt.title("D: %.2f, P-val: %.2g, N: %d (%.1f%%)" %
                  (ks[0], ks[1], N, frac))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/%s_%s_ks_validation.png" %
                    (self.plot_path, prefix, cctype))

        with open("%s/%s_%s_ks_validation_D.tsv" % (self.plot_path, prefix, cctype), "w") as f:
            for i in range(len(D)):
                f.write("%f\t%f\n" % (D[i], cD[i]))

        with open("%s/%s_%s_ks_validation_S.tsv" % (self.plot_path, prefix, cctype), "w") as f:
            for i in range(len(S)):
                f.write("%f\t%f\n" % (S[i], cS[i]))

        # ROC curve

        Scores = np.concatenate((-S, -D))
        Truth = np.array(len(S) * [1] + len(D) * [0])

        auc_score = self.roc_curve_plot(Truth, Scores, cctype, prefix, N, frac)
        plt.close('All')

        return ks, auc_score, frac

    def roc_curve_plot(self, thruth, scores, cctype, prefix, N=None, frac=None):

        plt.figure(figsize=(4, 4), dpi=600)
        fig = plt.subplot(111)

        auc_score = roc_auc_score(thruth, scores)

        fpr, tpr, thr = roc_curve(thruth, scores)

        plt.plot([0, 1], [0, 1], color=self.color, linestyle="--")
        plt.plot(fpr, tpr, color=self.color, linestyle="-", lw=2)

        plt.grid(linestyle="-.", color=self.color, lw=0.3)

        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

        plt.title("AUC: %.2f, N: %d (%.1f%%)" % (auc_score, N, frac))

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/%s_%s_auc_validation.png" %
                    (self.plot_path, prefix, cctype))

        with open("%s/%s_%s_auc_validation.tsv" % (self.plot_path, prefix, cctype), "w") as f:
            for i in range(len(fpr)):
                f.write("%f\t%f\n" % (fpr[i], tpr[i]))

        return auc_score

    # Matrix plot

    def matrix_plot(self, data_path):

        sns.set_style("white")

        with h5py.File(data_path, "r") as hf:
            Mols = len(hf["keys"])
            Vars = len(hf["V"][0])

        plt.figure(figsize=(4, 4), dpi=300)

        fig = plt.subplot(111)

        ax = plt.gca()

        xmax = 3
        ymax = 6

        plt.xlim(0, xmax)
        plt.ylim(0, ymax)

        for v1 in range(1, xmax):
            plt.plot([v1, v1], [0, ymax], "-.", lw=0.5, color=self.color)
        for h1 in range(1, ymax):
            plt.plot([0, xmax], [h1, h1], "-.", lw=0.5, color=self.color)

        ax.add_patch(patches.Rectangle((0, 0), math.log10(
            Vars), math.log10(Mols), color=self.color))

        plt.yticks([t for t in range(7)])
        plt.xticks([t for t in range(4)])

        ax.set_xlabel("Latent variables (log10)")
        ax.set_ylabel("Molecules (log10)")

        fig.axes.spines["bottom"].set_color(self.color)
        fig.axes.spines["top"].set_color(self.color)
        fig.axes.spines["right"].set_color(self.color)
        fig.axes.spines["left"].set_color(self.color)

        fig.patch.set_facecolor("white")

        plt.savefig("%s/matrix_plot.png" % self.plot_path)

    # Projection plot

    def datashader_projection(self, proj, **kwargs):
        import datashader as ds
        import datashader.transfer_functions as tf

        def make_cmap(colors, position=None, bit=False):
            bit_rgb = np.linspace(0, 1, 256)
            if position is None:
                position = np.linspace(0, 1, len(colors))
            else:
                if len(position) != len(colors):
                    sys.exit("position length must be the same as colors")
                elif position[0] != 0 or position[-1] != 1:
                    sys.exit("position must start with 0 and end with 1")
            if bit:
                for i in range(len(colors)):
                    colors[i] = (bit_rgb[colors[i][0]],
                                 bit_rgb[colors[i][1]],
                                 bit_rgb[colors[i][2]])
            cdict = {'red': [], 'green': [], 'blue': []}
            for pos, color in zip(position, colors):
                cdict['red'].append((pos, color[0], color[0]))
                cdict['green'].append((pos, color[1], color[1]))
                cdict['blue'].append((pos, color[2], color[2]))

            cmap = matplotlib.colors.LinearSegmentedColormap(
                'my_colormap', cdict, 256)
            return cmap

        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            return tuple(int(value[i:i + int(lv / 3)], 16) for i in range(0, lv, int(lv / 3)))

        def get_cmap(color):
            gray = self._rgb2hex(220, 218, 219)
            white = self._rgb2hex(250, 250, 250)
            black = self._rgb2hex(0, 0, 0)
            colors = [hex_to_rgb(c) for c in [black, white]]
            return make_cmap(colors, bit=True)

        # get kwargs
        cmap = kwargs.get('cmap', None)
        noise_scale = kwargs.get('noise_scale', None)
        how = kwargs.get('how', 'log')
        plot_size = kwargs.get('plot_size', (1000, 1000))
        x_range = kwargs.get('x_range', (-100, 100))
        y_range = kwargs.get('y_range', (-100, 100))
        spread = kwargs.get('spread', None)
        spread_threshold = kwargs.get('spread_threshold', None)
        weigth = kwargs.get('weigth', None)
        transparent = kwargs.get('transparent', False)
        marginals = kwargs.get('marginals', False)
        small = kwargs.get('small', True)
        category = kwargs.get('category', None)
        category_colors = kwargs.get('category_colors', None)
        save_each = kwargs.get('save_each', False)
        name = kwargs.get('name', 'PROJ')
        span = kwargs.get('span', None)
        if cmap is None:
            cmap = get_cmap(self.color)
        else:
            cmap = plt.cm.get_cmap(cmap)
        proj = proj[:]
        # apply noise?
        if noise_scale is None:
            noise = np.zeros_like(proj)
        else:
            noise = np.random.normal(size=proj.shape) / noise_scale
        # weight or categories?
        if weigth is not None:
            if not isinstance(weigth, list):
                weigth = [weigth] * len(proj)
            df = pd.DataFrame(
                data=np.hstack((proj + noise, np.expand_dims(weigth, 1))),
                columns=['x', 'y', 'w'])
        elif category is not None:
            if not len(category) == len(proj):
                raise Exception("Category list must have same" +
                                " dimensions as projection.")
            df = pd.DataFrame(
                data=np.hstack((proj + noise, np.expand_dims(category, 1))),
                columns=['x', 'y', 'cat'])
            df['cat'] = df['cat'].astype('category')
        else:
            df = pd.DataFrame(data=proj + noise, columns=['x', 'y'])
        # set canvas
        plot_height, plot_width = plot_size
        canvas = ds.Canvas(plot_height=plot_height, plot_width=plot_width,
                           x_range=x_range, y_range=y_range,
                           x_axis_type='linear', y_axis_type='linear')
        # aggregate
        if weigth is not None:
            points = canvas.points(df, 'x', 'y', ds.mean('w'))
        elif category is not None:
            points = canvas.points(df, 'x', 'y', ds.count_cat('cat'))
        else:
            points = canvas.points(df, 'x', 'y')
        # shading
        if category_colors:
            shade = tf.shade(points, color_key=category_colors, how=how,
                             span=span)
        else:
            shade = tf.shade(points, cmap=cmap, how=how, span=span)
        if spread is not None:
            if spread_threshold is None:
                shade = tf.spread(shade, px=spread)
            else:
                shade = tf.dynspread(
                    shade, threshold=spread_threshold, max_px=spread)
        if transparent:
            img = shade
        else:
            background_color = kwargs.get('background_color',
                                          self.cc_colors(self.dataset_code, 0))
            img = tf.set_background(shade, background_color)
        # export
        dst_file = os.path.join(self.plot_path, 'shaded_%s_%s' %
                                (name, self.dataset_code))
        ds.utils.export_image(img=img, filename=dst_file, fmt=".png")

        # save each category
        if category is not None and save_each:
            for cat, col in zip(np.unique(category), category_colors):
                cat_points = points.sel(cat=cat)
                shade = tf.shade(cat_points,
                                 cmap=[col], how=how)
                if spread is not None:
                    shade = tf.spread(shade, px=spread)
                if transparent:
                    img = shade
                else:
                    # background_color = kwargs.get('background_color',
                    #                              self.color)
                    img = tf.set_background(shade, background_color)
                dst_file = os.path.join(self.plot_path, 'shaded_%s_%s_%s' %
                                        (name, self.dataset_code, cat))
                ds.utils.export_image(img=img, filename=dst_file, fmt=".png")

        if marginals:
            f = plt.figure(figsize=(10, 10))
            gs = plt.GridSpec(6, 6)

            ax_joint = f.add_subplot(gs[1:, :-1])
            ax_marg_x = f.add_subplot(gs[0, :-1])
            ax_marg_y = f.add_subplot(gs[1:, -1])

            # Turn off tick visibility for the measure axis on the marginal
            # plots
            plt.setp(ax_marg_x.get_xticklabels(), visible=False)
            plt.setp(ax_marg_y.get_yticklabels(), visible=False)

            # Turn off the ticks on the density axis for the marginal plots
            plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_x.get_yticklabels(), visible=False)
            plt.setp(ax_marg_y.get_xticklabels(), visible=False)
            plt.setp(ax_joint.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax_joint.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax_joint.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax_joint.yaxis.get_minorticklines(), visible=False)
            plt.setp(ax_joint.get_yticklabels(), visible=False)
            plt.setp(ax_joint.get_xticklabels(), visible=False)

            ax_marg_x.yaxis.grid(False)
            ax_marg_x.set_xlim(x_range)
            ax_marg_y.xaxis.grid(False)
            ax_marg_y.set_ylim(x_range)

            sns.despine(ax=ax_marg_x, left=True)
            sns.despine(ax=ax_marg_y, bottom=True)
            f.tight_layout()
            f.subplots_adjust(hspace=0, wspace=0)

            color = plt.cm.get_cmap(cmap)(0)
            sns.kdeplot(proj[:, 0], ax=ax_marg_x, shade=True, color=color)
            sns.kdeplot(proj[:, 1], ax=ax_marg_y, vertical=True, shade=True,
                        color=color)

            with open(dst_file + ".png", 'r') as image_file:
                image = plt.imread(image_file)
                ax_joint.imshow(image)
            # g.ax_joint.set_axis_off()
            #sns.despine(top=True, right=True, left=True, bottom=True)
            dst_file = os.path.join(self.plot_path, 'shaded_margin_%s_%s.png' %
                                    (name, self.dataset_code))

            plt.savefig(dst_file, dpi=100, transparent=True)

        if small:
            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)
            values.pop('frame', None)
            values.pop('self', None)
            args = dict(values)
            args.update({'plot_size': (200, 200)})
            args.update({'small': False})
            args.update({'name': name + "_small"})
            self.datashader_projection(**args)

        return df

    def projection_plot(self, Proj, bw=None, levels=5, dev=None, s=None, transparency=0.5, **kwargs):

        if dev:
            noise_x = np.random.normal(0, dev, Proj.shape[0])
            noise_y = np.random.normal(0, dev, Proj.shape[0])

        gray = self._rgb2hex(220, 218, 219)

        if not s:
            s = np.max([0.3, -4e-6 * Proj.shape[0] + 4])

        fig = plt.figure(figsize=(5, 5), dpi=600)
        ax = fig.add_subplot(111)

        ax.set_facecolor(self.color)
        if dev:
            X = Proj[:, 0] + noise_x
            Y = Proj[:, 1] + noise_y
        else:
            X = Proj[:, 0]
            Y = Proj[:, 1]
        ax.scatter(X, Y, alpha=transparency, s=s, color="white")
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        name = kwargs.get('name', 'PROJ')
        dst_file = os.path.join(self.plot_path, 'shaded_margin_%s_%s.png' %
                                (name, self.dataset_code))
        plt.savefig(dst_file)

        def make_cmap(colors, position=None, bit=False):
            bit_rgb = np.linspace(0, 1, 256)
            if position is None:
                position = np.linspace(0, 1, len(colors))
            else:
                if len(position) != len(colors):
                    sys.exit("position length must be the same as colors")
                elif position[0] != 0 or position[-1] != 1:
                    sys.exit("position must start with 0 and end with 1")
            if bit:
                for i in range(len(colors)):
                    colors[i] = (bit_rgb[colors[i][0]],
                                 bit_rgb[colors[i][1]],
                                 bit_rgb[colors[i][2]])
            cdict = {'red': [], 'green': [], 'blue': []}
            for pos, color in zip(position, colors):
                cdict['red'].append((pos, color[0], color[0]))
                cdict['green'].append((pos, color[1], color[1]))
                cdict['blue'].append((pos, color[2], color[2]))

            cmap = matplotlib.colors.LinearSegmentedColormap(
                'my_colormap', cdict, 256)
            return cmap

        def hex_to_rgb(value):
            value = value.lstrip('#')
            lv = len(value)
            return tuple(int(value[i:i + int(lv / 3)], 16) for i in range(0, lv, int(lv / 3)))

        def get_cmap(color):
            colors = [hex_to_rgb(c) for c in [color, gray]]
            return make_cmap(colors, bit=True)

        sns.set_style("white")

        cmap = get_cmap(self.color)

        xmin, xmax = np.min(Proj[:, 0]), np.max(Proj[:, 0])
        ymin, ymax = np.min(Proj[:, 1]), np.max(Proj[:, 1])

        margin = 0.1

        xran = xmax - xmin
        xmin = xmin - margin * xran
        xmax = xmax + margin * xran
        yran = ymax - ymin
        ymin = ymin - margin * yran
        ymax = ymax + margin * yran

        # Peform the kernel density estimate
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        Proj = np.vstack( [ tuple(row) for row in Proj ] )
        values = np.vstack([Proj[:, 0], Proj[:, 1]])
        kernel = gaussian_kde(values, bw_method=bw)
        f = np.reshape(kernel(positions).T, xx.shape)

        # Plot
        fig = plt.figure(figsize=(5, 5), dpi=600)

        ax = fig.add_subplot(111)

        cut = np.percentile(f, 95)

        f[f > cut] = cut
        levels = np.linspace(0, cut, num=levels)
        f[f == np.min(f)] = 0

        ax.grid(False)
        plt.contourf(xx, yy, f, list(levels) +
                     [levels[-1] + 5 * (levels[-1] - levels[-2])], cmap=cmap)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        ax.axes.spines["bottom"].set_color(self.color)
        ax.axes.spines["top"].set_color(self.color)
        ax.axes.spines["right"].set_color(self.color)
        ax.axes.spines["left"].set_color(self.color)

        plt.savefig("%s/largevis.png" % self.plot_path,
                    bbox_inches="tight", pad_inches=0)

        xlim = ax.axes.get_xlim()
        ylim = ax.axes.get_ylim()

        return xlim, ylim

    def projection_plot_other(self, reference, library, bw_bg=0.2, bw_lib=0.15,
                              perc_bg=95, perc_lib=99, jitter=True):

        def projection_plot_preprocess(reference, bw, levels, lim=None, perc=95):
            if lim is None:
                margin = 0.1
                xmin, xmax = np.min(reference[:, 0]), np.max(reference[:, 0])
                ymin, ymax = np.min(reference[:, 1]), np.max(reference[:, 1])
                xran = xmax - xmin
                xmin = xmin - margin * xran
                xmax = xmax + margin * xran
                yran = ymax - ymin
                ymin = ymin - margin * yran
                ymax = ymax + margin * yran
            else:
                xmin, xmax, ymin, ymax = lim
            lim = (xmin, xmax, ymin, ymax)
            # Peform the kernel density estimate
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            reference = np.vstack({tuple(row) for row in reference})
            values = np.vstack([reference[:, 0], reference[:, 1]])
            kernel = gaussian_kde(values, bw_method=bw)
            f = np.reshape(kernel(positions).T, xx.shape)
            # Plot
            cut = np.percentile(f, perc)
            f[f > cut] = cut
            levels = np.linspace(0, cut, num=levels)
            f[f == np.min(f)] = 0
            return xx, yy, f, list(levels), lim

        def projection_plot(ax, reference, X, bw_bg=0.2, bw_lib=0.1,
                            levels_=15, title=None, perc_bg=95, perc_lib=99):
            sns.set_style("white")
            xx_1, yy_1, f_1, levels_1, lim = projection_plot_preprocess(
                reference, bw_bg, levels_, perc=perc_bg)
            ax.contour(xx_1, yy_1, f_1, levels_1,
                       colors="white", linewidths=0.5)
            xx_2, yy_2, f_2, levels_2, _ = projection_plot_preprocess(
                X, bw_lib, levels_, lim, perc=perc_lib)
            ax.contourf(xx_2, yy_2, f_2, levels_2, cmap="Spectral_r")
            ax.scatter(X[:, 0], X[:, 1], color="black", s=0.2)
            ax.grid(False)
            xmin, xmax, ymin, ymax = lim
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            if title:
                ax.set_title(title, fontname="Courier New", fontsize=20)
            return xx_1, yy_1, f_1, levels_1, lim

        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        if jitter:
            noise = np.random.normal(0, .5, library.shape)
            lib = library + noise
        else:
            lib = library
        projection_plot(ax, reference, lib, bw_bg=bw_bg,
                        bw_lib=bw_lib, perc_bg=perc_bg, perc_lib=perc_lib)
        filename = os.path.join(self.plot_path, "projections.png")
        plt.savefig(filename, dpi=200)
        plt.close()

    def sign_feature_distribution_plot(self, sign, max_samples=10000):
        if sign.shape[0] > max_samples:
            keys = np.random.choice(sign.keys, max_samples, replace=False)
            matrix = sign.get_vectors(keys)[1]
        else:
            matrix = sign.get_h5_dataset('V')
        df = pd.DataFrame(matrix).melt()

        coord = self.dataset_code
        fig = plt.figure(figsize=(10, 3), dpi=100)
        ax = fig.add_subplot(111)
        sns.pointplot(x='variable', y='value', data=df,
                      ax=ax, errorbar='sd', linestyle='none', markers='.',
                      color=self._coord_color(coord))
        ax.set_ylim(-2, 2)
        ax.set_xlim(-2, 130)
        ax.set_xticks([])
        ax.set_xlabel('')
        min_mean = min(np.mean(matrix, axis=0))
        max_mean = max(np.mean(matrix, axis=0))
        ax.fill_between([-2, 130], [max_mean, max_mean], [min_mean, min_mean],
                        facecolor=self._coord_color(coord), alpha=0.4,
                        zorder=0)
        sns.despine(bottom=True)
        filename = os.path.join(self.plot_path, "feat_distrib_%s.png" % coord)
        plt.savefig(filename, dpi=100)
        plt.close()

    def quick_sign_pred_vs_true_scatter(self, y_true, y_pred):

        def quick_gaussian_kde(x, y, limit=5000):
            xl = x[:limit]
            yl = y[:limit]
            xy = np.vstack([xl, yl])
            c = gaussian_kde(xy)(xy)
            order = c.argsort()
            return xl, yl, c, order

        fig = plt.figure(figsize=(40, 80))
        gs1 = gridspec.GridSpec(16, 8)
        gs1.update(wspace=0.0, hspace=0.0)  # set the spacing between axes.
        for i in range(128):
            # i = i + 1 # grid spec indexes from 0
            ax = plt.subplot(gs1[i])
            plt.axis('on')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            x, y, c, order = quick_gaussian_kde(y_true[:, i], y_pred[:, i])
            ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
        fig.tight_layout()
        filename = os.path.join(
            self.plot_path, "pred_vs_true.png")
        plt.savefig(filename, dpi=60)
        plt.close()

    @skip_on_exception
    def sign2_prediction_plot(self, y_true, y_pred, predictor_name, max_samples=1000):

        coord = self.dataset_code
        self.__log.info("sign2 %s predicted vs. actual %s",
                        coord, predictor_name)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        # we are assuming a dimension of 128 components
        f, axarr = plt.subplots(
            8, 16, sharex=True, sharey=True, figsize=(60, 30))
        for comp, ax in zip(range(128), axarr.flatten()):
            # for plotting we subsample randomly
            nr_samples = len(y_true[:, comp])
            if nr_samples > max_samples:
                mask = np.random.choice(
                    range(nr_samples), max_samples, replace=False)
                x = y_true[:, comp][mask]
                y = y_pred[:, comp][mask]
            else:
                x = y_true[:, comp]
                y = y_pred[:, comp]
            rsquare = r2_score(x, y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x, y)
            ax.scatter(x, y, color=self._coord_color(coord), marker='.',
                       alpha=0.3, zorder=2)
            # add lines
            ax.plot([min_val, max_val],
                    [min_val * slope + intercept, max_val * slope + intercept],
                    color=self._coord_color(coord))
            ax.plot([min_val, max_val], [min_val, max_val], '--',
                    color=self._coord_color(coord))
            ax.axhline(0, min_val, max_val, ls='--',
                       lw=0.5, color='grey', zorder=1)
            ax.axvline(0, min_val, max_val, ls='--',
                       lw=0.5, color='grey', zorder=1)
            # fix limits considering all components dynamic range
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            # add R square stat
            cmap = plt.cm.get_cmap('jet_r')
            txt = ax.text(0.05, 0.85, "$R^2$: {:.2f}".format(rsquare),
                          transform=ax.transAxes,
                          size=30, color=cmap(rsquare / 2))
            txt.set_path_effects([path_effects.Stroke(
                linewidth=1, foreground='black'), path_effects.Normal()])
            # visualize linear regression equation
            ax.text(0.2, 0.05,
                    "$y = {:.2f}x {:+.2f}$".format(slope, intercept),
                    transform=ax.transAxes,
                    size=20, color='k')
            """
            # following would make one plot for each component along with
            # distributions (takes too long)
            g = sns.JointGrid(x=y_true[:, comp], y=y_pred[:, comp], space=0)
            g = g.plot_joint(sns.regplot, color="r")
            g = g.plot_joint(sns.kdeplot, cmap="Greys")
            g = g.plot_marginals(sns.kdeplot, color="r", shade=True)
            rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
            g = g.annotate(rsquare, template="{stat}: {val:.2f}",
                           stat="$R^2$", loc="upper left", fontsize=12)
            plt.savefig('pred_vs_true_%s_%s.png' % (coord, comp), dpi=100)
            """
        f.tight_layout()
        filename = os.path.join(
            self.plot_path, "pred_vs_true_%s.png" % predictor_name)
        plt.savefig(filename, dpi=60)
        plt.close()

        ####
        # relationship between predicted and observed distances
        ####
        self.__log.info("sign2 %s DISTANCES predicted vs. actual %s",
                        coord, predictor_name)

        # for plotting we subsample randomly
        nr_samples = len(y_true)
        if nr_samples > 1000:
            mask = np.random.choice(
                range(nr_samples), 1000, replace=False)
            x = cosine_distances(y_true[mask])[np.tril_indices(1000, -1)]
            y = cosine_distances(y_pred[mask])[np.tril_indices(1000, -1)]
        else:
            x = cosine_distances(y_true)[np.tril_indices(nr_samples, -1)]
            y = cosine_distances(y_pred)[np.tril_indices(nr_samples, -1)]
        # stats
        pearson = stats.pearsonr(x, y)[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            x, y)
        # plot
        f, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(x, y, color=self._coord_color(coord), marker='.',
                   alpha=0.2, zorder=2)
        # add lines
        min_val = 0.0
        max_val = 1.0
        ax.plot([min_val, max_val],
                [min_val * slope + intercept, max_val * slope + intercept],
                color=self._coord_color(coord))
        ax.plot([min_val, max_val], [min_val, max_val], '--',
                color=self._coord_color(coord))
        ax.axhline(0, min_val, max_val, ls='--',
                   lw=0.5, color='grey', zorder=1)
        ax.axvline(0, min_val, max_val, ls='--',
                   lw=0.5, color='grey', zorder=1)
        # fix limits of distance metric
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # set labels
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Distances")
        # add R square stat
        cmap = plt.cm.get_cmap('jet_r')
        txt = ax.text(0.05, 0.85, "$\\rho$: {:.2f}".format(pearson),
                      transform=ax.transAxes,
                      size=30, color=cmap(pearson / 2))
        txt.set_path_effects([path_effects.Stroke(
            linewidth=1, foreground='black'), path_effects.Normal()])
        # visualize linear regression equation
        ax.text(0.2, 0.05,
                "$y = {:.2f}x {:+.2f}$".format(slope, intercept),
                transform=ax.transAxes,
                size=20, color='k')

        f.tight_layout()
        filename = os.path.join(
            self.plot_path, "DISTANCES_%s.png" % predictor_name)
        plt.savefig(filename, dpi=60)
        plt.close()

        self.__log.info("sign2 %s DISTANCES KDE predicted vs. actual %s",
                        coord, predictor_name)
        if len(x) > 1000:
            mask = np.random.choice(range(len(x)), 1000, replace=False)
            x = x[mask]
            y = y[mask]
        dfplot = pd.DataFrame()
        dfplot['x'] = x
        dfplot['y'] = y
        sns.jointplot(data=dfplot,  x='x', y='y', kind="kde", color=self._coord_color(coord),
                      height=7, xlim=(0, 1), ylim=(0, 1))
        filename = os.path.join(
            self.plot_path, "DISTANCES_kde_%s.png" % predictor_name)
        plt.savefig(filename, dpi=60)
        plt.close()

    @staticmethod
    def _get_grid_search_df(grid_root):
        dir_names = [name for name in os.listdir(
            grid_root) if os.path.isdir(os.path.join(grid_root, name))]
        for dir_name in dir_names:
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in dir_name.split("-")}
            tmpdf_file = os.path.join(grid_root, dir_name, 'stats.pkl')
            if os.path.isfile(tmpdf_file):
                break
        cols = list(pd.read_pickle(tmpdf_file).columns)
        df = pd.DataFrame(columns=set(cols) | set(params.keys()))
        for dir_name in dir_names:
            tmpdf_file = os.path.join(grid_root, dir_name, 'stats.pkl')
            if not os.path.isfile(tmpdf_file):
                print("File not found: %s", tmpdf_file)
                continue
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in dir_name.split("-")}
            tmpdf = pd.read_pickle(tmpdf_file)
            for k, v in params.items():
                tmpdf[k] = pd.Series([v] * len(tmpdf))
            df = pd.concat([ df, tmpdf ], ignore_index=True)
        return df

    def sign2_grid_search_plot(self, grid_root):
        dir_names = [name for name in os.listdir(
            grid_root) if os.path.isdir(os.path.join(grid_root, name))]
        for dir_name in dir_names:
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in dir_name.split("-")}
            tmpdf_file = os.path.join(grid_root, dir_name, 'stats.pkl')
            if os.path.isfile(tmpdf_file):
                break
        cols = list(pd.read_pickle(tmpdf_file).columns)
        df = pd.DataFrame(columns=set(cols) | set(params.keys()))
        for dir_name in dir_names:
            tmpdf_file = os.path.join(grid_root, dir_name, 'stats.pkl')
            if not os.path.isfile(tmpdf_file):
                print("File not found: %s", tmpdf_file)
                continue
            params = {n.rsplit("_", 1)[0]: n.rsplit("_", 1)[1]
                      for n in dir_name.split("-")}
            tmpdf = pd.read_pickle(tmpdf_file)
            for k, v in params.items():
                tmpdf[k] = pd.Series([v] * len(tmpdf))
            df = pd.concat([ df, tmpdf ], ignore_index=True)

        df['layer_size'] = df['layer_size'].astype(int)
        df['adanet_iterations'] = df['adanet_iterations'].astype(int)
        df['adanet_lambda'] = df['adanet_lambda'].astype(float)

        for metric in ['pearson_avg', 'nr_variables']:
            ada_df = df[df.algo == 'AdaNet']
            hue_order = sorted(list(ada_df.subnetwork_generator.unique()))
            g = sns.relplot(y=metric, style="dataset", hue='subnetwork_generator',
                            x='adanet_lambda', col='adanet_iterations',
                            hue_order=hue_order,
                            kind='scatter', data=ada_df)
            # linreg
            linreg_train = df[(df.algo != 'AdaNet') & (
                df.dataset == 'train')].iloc[0][metric]
            linreg_test = df[(df.algo != 'AdaNet') & (
                df.dataset == 'test')].iloc[0][metric]
            for ax in g.axes.flatten():
                ax.axhline(linreg_train, ls='--',
                           lw=0.5, color='grey', zorder=1)
                ax.axhline(linreg_test,
                           lw=0.5, color='grey', zorder=1)

            filename = os.path.join(
                self.plot_path, "sign2_%s_grid_search_%s.png" % (self.dataset_code, metric))
            plt.savefig(filename, dpi=100)
            plt.close()

    def sign3_adanet_comparison(self, sign3, metric="pearson", pathbase="adanet", hue_order=None, palette=None, ds_name=None, filter_from=None):
        dir_names = list()
        for name in os.listdir(sign3.model_path):
            if pathbase is not None and not name.startswith(pathbase):
                continue
            if os.path.isdir(os.path.join(sign3.model_path, name)):
                self.__log.debug(name)
                dir_names.append(name)
        for dir_name in dir_names:
            tmpdf_file = os.path.join(sign3.model_path, dir_name, 'stats.pkl')
            if os.path.isfile(tmpdf_file):
                cols = list(pd.read_pickle(tmpdf_file).columns)
                break
        df = pd.DataFrame(columns=set(cols))
        for dir_name in dir_names:
            if pathbase is not None and not dir_name.startswith(pathbase):
                continue
            tmpdf_file = os.path.join(sign3.model_path, dir_name, 'stats.pkl')
            if not os.path.isfile(tmpdf_file):
                continue
            tmpdf = pd.read_pickle(tmpdf_file)
            self.__log.debug("%s lines in: %s", len(tmpdf), dir_name)
            if ds_name:
                tmpdf = tmpdf.replace("-self", "not-%s" % ds_name)
            df = pd.concat([ df, tmpdf], ignore_index=True)

        if filter_from:
            df = df[df['from'].isin(filter_from)]

        froms = sorted(list(df['from'].unique()))
        order = froms
        if 'ALL' in froms:
            froms.remove('ALL')
            order = ['ALL'] + froms
        if hue_order is None:
            hue_order = sorted(df[df['from'] == 'A1.001']['algo'].unique())
        else:
            df = df[df.algo.isin(hue_order)]
        if palette is None:
            palette = sns.color_palette("Blues")

        sns.set_style("whitegrid")
        g = sns.catplot(x="from", y=metric, hue="algo", row="dataset",
                        order=order, hue_order=hue_order,
                        row_order=['train', 'test', 'validation'],
                        legend=False, sharex=False,
                        data=df, height=6, aspect=3, kind="bar",
                        palette=palette)
        g.map_dataframe(sns.stripplot, x="from", y="coverage",
                        order=order, jitter=False, palette=['crimson'])
        plt.legend(loc='upper right')
        if metric == "pearson":
            g.set(ylim=(0, 1))
        if pathbase is not None:
            qual = "_".join([self.dataset_code, metric, pathbase])
        else:
            qual = "_".join([self.dataset_code, metric])
        if ds_name:
            qual += "_%s" % ds_name
        filename = os.path.join(self.plot_path, "sign3_%s.png" % qual)
        plt.savefig(filename, dpi=100)
        plt.close()

    def sign3_crosspred_comparison(self, sign3, metric="pearson", pathbase="crosspred"):
        dir_names = list()
        for name in os.listdir(sign3.model_path):
            if pathbase is not None and not name.startswith(pathbase):
                continue
            if os.path.isdir(os.path.join(sign3.model_path, name)):
                dir_names.append(name)
        for dir_name in dir_names:
            tmpdf_file = os.path.join(sign3.model_path, dir_name, 'stats.pkl')
            if os.path.isfile(tmpdf_file):
                cols = list(pd.read_pickle(tmpdf_file).columns)
                break
        df = pd.DataFrame(columns=set(cols) | {'from'})
        for dir_name in dir_names:
            if pathbase is not None and not dir_name.startswith(pathbase):
                continue
            tmpdf_file = os.path.join(sign3.model_path, dir_name, 'stats.pkl')
            if not os.path.isfile(tmpdf_file):
                continue
            tmpdf = pd.read_pickle(tmpdf_file)
            self.__log.debug("%s lines in: %s", len(tmpdf), dir_name)
            if 'from' not in tmpdf:
                from_ds = dir_name.split("_")[1]
                tmpdf['from'] = pd.Series([from_ds] * len(tmpdf))
            df = pd.concat([ df, tmpdf], ignore_index=True)

        froms = sorted(list(df['from'].unique()))
        order = froms
        if 'ALL' in froms:
            froms.remove('ALL')
            order = ['ALL'] + froms
        if 'AVG' in froms:
            froms.remove('AVG')
            order = ['AVG'] + froms
        hue_order = df[(df.dataset == 'test') & (
            df['from'] == 'A1.001')].sort_values(metric)['algo'].to_list()

        sns.set_style("whitegrid")
        g = sns.catplot(x="from", y=metric, hue="algo", row="dataset",
                        order=order, hue_order=hue_order,
                        row_order=['train', 'test', 'validation'],
                        legend_out=False, sharex=False,
                        data=df, height=6, aspect=3, kind="bar",
                        palette=[sns.color_palette('Greens')[2],
                                 sns.color_palette('Oranges')[2],
                                 sns.color_palette("Blues")[3]])
        g.set(ylim=(0, 1))
        if pathbase is not None:
            qual = "_".join([self.dataset_code, metric, pathbase])
        else:
            qual = "_".join([self.dataset_code, metric])
        filename = os.path.join(self.plot_path, "sign3_%s.png" % qual)
        plt.savefig(filename, dpi=100)
        plt.close()

    def sign3_confidences(self, sign3, suffix=None, skip_exp_error=True):

        # load data
        self.__log.info("loading data")
        error_file = os.path.join(sign3.model_path, 'error.h5')
        with h5py.File(error_file, "r") as hf:
            keys = hf['keys'][:]
            train_log_mse = hf['log_mse_consensus'][:]
            train_log_mse_real = hf['log_mse'][:]
            self.__log.info("train_log_mse %s", train_log_mse.shape)
        # test is anything that wasn't in the confidence distribution
        test_keys = list(sign3.unique_keys - set(keys))
        test_idxs = np.where(np.isin(list(sign3.keys), test_keys))[0]
        train_idxs = np.where(~np.isin(list(sign3.keys), test_keys))[0]

        confidence = sign3.get_h5_dataset('confidence')
        test_confidence = confidence[test_idxs]
        self.__log.info("test_confidence %s", test_confidence.shape)
        train_confidence = confidence[train_idxs]
        self.__log.info("train_confidence %s", train_confidence.shape)

        stddev = sign3.get_h5_dataset('stddev')
        test_std = stddev[test_idxs]
        self.__log.info("test_std %s", test_std.shape)
        train_std = stddev[train_idxs]
        self.__log.info("train_std %s", train_std.shape)
        stddev_norm = sign3.get_h5_dataset('stddev_norm')
        test_std_norm = stddev_norm[test_idxs]
        self.__log.info("test_std_norm %s", test_std_norm.shape)
        train_std_norm = stddev_norm[train_idxs]
        self.__log.info("train_std_norm %s", train_std_norm.shape)

        intensity = sign3.get_h5_dataset('intensity')
        test_inte = intensity[test_idxs]
        self.__log.info("test_inte %s", test_inte.shape)
        train_inte = intensity[train_idxs]
        self.__log.info("train_inte %s", train_inte.shape)
        intensity_norm = sign3.get_h5_dataset('intensity_norm')
        test_inte_norm = intensity_norm[test_idxs]
        self.__log.info("test_inte_norm %s", test_inte.shape)
        train_inte_norm = intensity_norm[train_idxs]
        self.__log.info("train_inte_norm %s", train_inte.shape)

        if not skip_exp_error:
            exp_error = sign3.get_h5_dataset('exp_error')
            test_err = exp_error[test_idxs]
            self.__log.info("test_err %s", test_err.shape)
            train_err = exp_error[train_idxs]
            self.__log.info("train_err %s", train_err.shape)
            exp_error_norm = sign3.get_h5_dataset('exp_error_norm')
            test_err_norm = exp_error_norm[test_idxs]
            self.__log.info("test_err %s", test_err.shape)
            train_err_norm = exp_error_norm[train_idxs]
            self.__log.info("train_err %s", train_err.shape)

        def quick_gaussian_kde(x, y, limit=10000):
            xl = x[:limit]
            yl = y[:limit]
            xy = np.vstack([xl, yl])
            c = gaussian_kde(xy)(xy)
            order = c.argsort()
            return xl, yl, c, order

        # prepare plot space
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(16, 16))
        grid = plt.GridSpec(6, 3)
        color = coord_color(self.dataset_code)

        # stddev row
        # train test distributions
        self.__log.info("plotting stddev distributions")
        ax = fig.add_subplot(grid[0, 0])
        sns.distplot(test_std, hist_kws={'range': (0, 0.3)},
                     ax=ax, kde=False, norm_hist=False, color=color)
        sns.distplot(train_std, hist_kws={'range': (0, 0.3)},
                     ax=ax, kde=False, norm_hist=False, color='grey')
        ax.set_xlabel('stddev')
        ax.set_xlim(0)
        ax.set_yscale('log')
        ax.set_ylabel('molecules')

        # train mse correlation
        self.__log.info("plotting stddev vs log mse")
        ax = fig.add_subplot(grid[0, 1])
        x, y, c, order = quick_gaussian_kde(train_std, train_log_mse)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('stddev')
        ax.set_ylabel('consensus log mse')
        pearson_coeff_stddev = stats.pearsonr(train_std, train_log_mse)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff_stddev),
                      transform=ax.transAxes, size=10)

        # test normalized distributions
        self.__log.info("plotting stddev norm distribution")
        ax = fig.add_subplot(grid[0, 2])
        sns.distplot(test_std_norm, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color=color)
        ax.set_xlabel('normalized stddev')
        ax.set_xlim(0)
        ax.set_yscale('log')
        ax.set_ylabel('molecules')

        # intensity row
        # train test distributions
        self.__log.info("plotting intensity distributions")
        ax = fig.add_subplot(grid[1, 0])
        sns.distplot(test_inte, hist_kws={'range': (0, 0.6)},
                     ax=ax, kde=False, norm_hist=False, color=color)
        sns.distplot(train_inte, hist_kws={'range': (0, 0.6)},
                     ax=ax, kde=False, norm_hist=False, color='grey')
        ax.set_xlabel('intensity')
        ax.set_xlim(0)
        ax.set_yscale('log')
        ax.set_ylabel('molecules')

        # train mse correlation
        self.__log.info("plotting intensity vs log mse")
        ax = fig.add_subplot(grid[1, 1])
        x, y, c, order = quick_gaussian_kde(train_inte, train_log_mse)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('intensity')
        ax.set_ylabel('consensus log mse')
        pearson_coeff_inte = stats.pearsonr(train_inte, train_log_mse)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff_inte),
                      transform=ax.transAxes, size=10)

        # test normalized distributions
        self.__log.info("plotting intensity norm distribution")
        ax = fig.add_subplot(grid[1, 2])
        sns.distplot(test_inte_norm, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color=color)
        ax.set_xlabel('normalized intensity')
        ax.set_xlim(0)
        ax.set_yscale('log')
        ax.set_ylabel('molecules')

        if not skip_exp_error:

            # pred_err row
            # train test distributions
            self.__log.info("plotting expected error distributions")
            ax = fig.add_subplot(grid[2, 0])
            sns.distplot(test_err, hist_kws={'range': (-3.5, -1)},
                         ax=ax, kde=False, norm_hist=False, color=color)
            sns.distplot(train_err, hist_kws={'range': (-3.5, -1)},
                         ax=ax, kde=False, norm_hist=False, color='grey')
            ax.set_xlabel('expected error')
            # ax.set_xlim(0)
            ax.set_yscale('log')
            ax.set_ylabel('molecules')

            # train mse correlation
            self.__log.info("plotting expected error vs log mse")
            ax = fig.add_subplot(grid[2, 1])
            x, y, c, order = quick_gaussian_kde(train_err, train_log_mse_real)
            ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
            ax.set_xlabel('expected error')
            ax.set_ylabel('log mse')
            pearson_coeff_err = stats.pearsonr(
                train_err, train_log_mse_real)[0]
            txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff_err),
                          transform=ax.transAxes, size=10)

            # test normalized distributions
            self.__log.info("plotting expected error norm distribution")
            ax = fig.add_subplot(grid[2, 2])
            sns.distplot(test_err_norm, hist_kws={'range': (0, 1)},
                         ax=ax, kde=False, norm_hist=False, color=color)
            ax.set_xlabel('normalized expected error')
            # ax.set_xlim(0)
            ax.set_yscale('log')
            ax.set_ylabel('molecules')

        # plot old confidence
        # prediction train test distributions
        self.__log.info("plotting old confidence distribution")
        ax = fig.add_subplot(grid[3, 0])
        test_confidence_old = (test_inte_norm * (1 - test_std_norm))**(1.0 / 2)
        train_confidence_old = (
            train_inte_norm * (1 - train_std_norm))**(1.0 / 2)
        sns.distplot(test_confidence_old, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color=color)
        sns.distplot(train_confidence_old, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color='grey')
        ax.set_xlabel('confidence old')
        ax.set_yscale('log')
        ax.set_ylabel('molecules')

        # predicted mse vs mse
        self.__log.info("plotting old confidence vs consensus mse")
        ax = fig.add_subplot(grid[3, 1])
        x, y, c, order = quick_gaussian_kde(
            train_confidence_old, train_log_mse)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('confidence old')
        ax.set_ylabel('consensus log mse')
        pearson_coeff = stats.pearsonr(train_confidence_old, train_log_mse)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff),
                      transform=ax.transAxes, size=10)

        self.__log.info("plotting old confidence vs true mse")
        ax = fig.add_subplot(grid[3, 2])
        x, y, c, order = quick_gaussian_kde(
            train_confidence_old, train_log_mse_real)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('confidence old')
        ax.set_ylabel('log mse')
        pearson_coeff = stats.pearsonr(
            train_confidence_old, train_log_mse_real)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff),
                      transform=ax.transAxes, size=10)

        # confidence row
        # prediction train test distributions
        self.__log.info("plotting confidence distribution")
        ax = fig.add_subplot(grid[4, 0])
        sns.distplot(test_confidence, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color=color)
        sns.distplot(train_confidence, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color='grey')
        ax.set_xlabel('confidence')
        ax.set_yscale('log')
        ax.set_ylabel('molecules')

        # predicted mse vs mse
        self.__log.info("plotting pred vs true mse")
        ax = fig.add_subplot(grid[4, 1])
        x, y, c, order = quick_gaussian_kde(train_confidence, train_log_mse)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('confidence')
        ax.set_ylabel('consensus log mse')
        pearson_coeff = stats.pearsonr(train_confidence, train_log_mse)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff),
                      transform=ax.transAxes, size=10)

        self.__log.info("plotting pred vs true mse")
        ax = fig.add_subplot(grid[4, 2])
        x, y, c, order = quick_gaussian_kde(
            train_confidence, train_log_mse_real)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('confidence')
        ax.set_ylabel('log mse')
        pearson_coeff = stats.pearsonr(train_confidence, train_log_mse_real)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff),
                      transform=ax.transAxes, size=10)

        # plot new confidence
        # prediction train test distributions
        self.__log.info("plotting confidence_test distribution")
        ax = fig.add_subplot(grid[5, 0])
        weights = [abs(pearson_coeff_inte), abs(
            pearson_coeff_stddev), abs(pearson_coeff_err)]
        test_confidence_new = np.average(
            [test_inte_norm, (1 - test_std_norm), (1 - test_err_norm)],
            weights=weights, axis=0)
        train_confidence_new = np.average(
            [train_inte_norm, (1 - train_std_norm), (1 - train_err_norm)],
            weights=weights, axis=0)

        sns.distplot(test_confidence_new, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color=color)
        sns.distplot(train_confidence_new, hist_kws={'range': (0, 1)},
                     ax=ax, kde=False, norm_hist=False, color='grey')
        ax.set_xlabel('confidence weighted')
        ax.set_yscale('log')
        ax.set_ylabel('molecules')

        # predicted mse vs mse
        self.__log.info("plotting test confidence vs consensus mse")
        ax = fig.add_subplot(grid[5, 1])
        x, y, c, order = quick_gaussian_kde(
            train_confidence_new, train_log_mse)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('confidence weighted')
        ax.set_ylabel('consensus log mse')
        pearson_coeff = stats.pearsonr(train_confidence_new, train_log_mse)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff),
                      transform=ax.transAxes, size=10)

        self.__log.info("plotting test confidence vs true mse")
        ax = fig.add_subplot(grid[5, 2])
        x, y, c, order = quick_gaussian_kde(
            train_confidence_new, train_log_mse_real)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_xlabel('confidence weighted')
        ax.set_ylabel('log mse')
        pearson_coeff = stats.pearsonr(
            train_confidence_new, train_log_mse_real)[0]
        txt = ax.text(0.85, 0.85, "p: {:.2f}".format(pearson_coeff),
                      transform=ax.transAxes, size=10)

        # save
        plt.tight_layout()
        if suffix is None:
            filename = os.path.join(
                self.plot_path, "sign3_error_%s.png" % self.dataset_code)
        else:
            filename = os.path.join(
                self.plot_path, "sign3_error_%s_%s.png" % (self.dataset_code, suffix))
        plt.savefig(filename, dpi=150)
        plt.close()

    def sign_component_correlation(self, sign, sample_size=10000):
        whole_sign = sign[:sample_size]
        corr = abs(1 - cdist(whole_sign.T, whole_sign.T, metric='correlation'))
        nr_vars = corr.shape[0]
        corr = corr[~np.eye(nr_vars, dtype=bool)].reshape(nr_vars, -1)

        df = pd.DataFrame(corr).melt()

        coord = self.dataset_code

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(10, 3), dpi=100)
        ax = fig.add_subplot(111)
        order = list(np.argsort(np.mean(corr, axis=0))[::-1])
        sns.boxplot(x='variable', y='value', data=df, order=order,
                    ax=ax, color=self._coord_color(coord),
                    linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_xlim(-2, 130)
        ax.set_xticks([])
        ax.set_xlabel('')
        filename = os.path.join(self.plot_path, "feat_corr_%s.png" % coord)
        plt.savefig(filename, dpi=100)
        plt.close()

    def exp_error(self, sign3, sign2_universe_presence):
        with h5py.File(sign2_universe_presence, 'r') as hf:
            presence = hf['V'][:]
        exp_error = sign3.get_h5_dataset('exp_error')
        datasets = [a + b for a in 'ABCDE' for b in '12345']
        df = pd.DataFrame(presence.astype(bool), columns=datasets)
        df['exp_error'] = exp_error

        sns.set_style("whitegrid")
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=False,
                                       figsize=(10, 10), dpi=100)

        known_df = df[df[sign3.dataset[:2]] == True]
        ax1.boxplot([known_df[known_df[ds] == True].exp_error.to_list()
                     for ds in datasets], labels=datasets)
        ax1.set_ylabel('pred log MSE')
        ax1.set_xlabel('with sign2 (%s)' % len(known_df))
        unkno_df = df[df[sign3.dataset[:2]] == False]
        ax2.boxplot([unkno_df[unkno_df[ds] == True].exp_error.to_list()
                     for ds in datasets], labels=datasets)
        ax2.set_ylabel('pred log MSE')
        ax2.set_xlabel('w/o  sign2 (%s)' % len(unkno_df))

        filename = os.path.join(self.plot_path,
                                "exp_error_%s.png" % sign3.dataset[:2])
        plt.savefig(filename, dpi=100)
        plt.close()

    def sign3_conf_scores(self, sign3):
                # load data
        self.__log.info("loading data")
        error_file = os.path.join(sign3.model_path, 'error.h5')
        with h5py.File(error_file, "r") as hf:
            keys = hf['keys'][:]
            train_log_mse = hf['log_mse_consensus'][:]
            train_log_mse_real = hf['log_mse'][:]
            self.__log.info("train_log_mse %s", train_log_mse.shape)
        # test is anything that wasn't in the confidence distribution
        test_keys = list(sign3.unique_keys - set(keys))
        test_idxs = np.where(np.isin(list(sign3.keys), test_keys))[0]
        train_idxs = np.where(~np.isin(list(sign3.keys), test_keys))[0]

        stddev = sign3.get_h5_dataset('stddev')
        test_std = stddev[test_idxs]
        self.__log.info("test_std %s", test_std.shape)
        train_std = stddev[train_idxs]
        self.__log.info("train_std %s", train_std.shape)
        stddev_norm = sign3.get_h5_dataset('stddev_norm')
        test_std_norm = stddev_norm[test_idxs]
        self.__log.info("test_std_norm %s", test_std_norm.shape)
        train_std_norm = stddev_norm[train_idxs]
        self.__log.info("train_std_norm %s", train_std_norm.shape)

        intensity = sign3.get_h5_dataset('intensity')
        test_inte = intensity[test_idxs]
        self.__log.info("test_inte %s", test_inte.shape)
        train_inte = intensity[train_idxs]
        self.__log.info("train_inte %s", train_inte.shape)
        intensity_norm = sign3.get_h5_dataset('intensity_norm')
        test_inte_norm = intensity_norm[test_idxs]
        self.__log.info("test_inte_norm %s", test_inte.shape)
        train_inte_norm = intensity_norm[train_idxs]
        self.__log.info("train_inte_norm %s", train_inte.shape)

        exp_error = sign3.get_h5_dataset('exp_error')
        test_err = exp_error[test_idxs]
        self.__log.info("test_err %s", test_err.shape)
        train_err = exp_error[train_idxs]
        self.__log.info("train_err %s", train_err.shape)
        exp_error_norm = sign3.get_h5_dataset('exp_error_norm')
        test_err_norm = exp_error_norm[test_idxs]
        self.__log.info("test_err %s", test_err.shape)
        train_err_norm = exp_error_norm[train_idxs]
        self.__log.info("train_err %s", train_err.shape)

        pearson_coeff_stddev = stats.pearsonr(train_std, train_log_mse)[0]
        pearson_coeff_inte = stats.pearsonr(train_inte, train_log_mse)[0]
        pearson_coeff_err = stats.pearsonr(train_err, train_log_mse_real)[0]

        # overall distributions (pairplot)
        stddev_norm = sign3.get_h5_dataset('stddev_norm')
        intensity_norm = sign3.get_h5_dataset('intensity_norm')
        experr_norm = sign3.get_h5_dataset('exp_error_norm')
        stddev = sign3.get_h5_dataset('stddev')
        intensity = sign3.get_h5_dataset('intensity')
        experr = sign3.get_h5_dataset('exp_error')
        s2_mask = sign3.get_h5_dataset('outlier') == 0

        weights = [abs(pearson_coeff_inte), abs(
            pearson_coeff_stddev), abs(pearson_coeff_err)]
        confidence_new = np.average(
            [intensity_norm, (1 - stddev_norm), (1 - experr_norm)],
            weights=weights, axis=0)
        novelty = np.log(abs(sign3.get_h5_dataset('novelty')))

        def quick_gaussian_kde(x, y, limit=20000):
            xl = x[:limit]
            yl = y[:limit]
            xy = np.vstack([xl, yl])
            c = gaussian_kde(xy)(xy)
            order = c.argsort()
            return xl, yl, c, order

        confidence_new = confidence_new[s2_mask]
        novelty = novelty[s2_mask]
        stddev = stddev[s2_mask]
        intensity = intensity[s2_mask]
        experr = experr[s2_mask]
        # prepare plot space
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(25, 5))
        grid = plt.GridSpec(1, 4)
        color = coord_color(self.dataset_code)

        ax = fig.add_subplot(grid[0, 0])
        x, y, c, order = quick_gaussian_kde(confidence_new, novelty)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_ylabel('log novelty')
        ax.set_ylim(0, np.percentile(novelty, 90))
        ax.set_xlabel('confidence weigthed')
        pc = stats.pearsonr(novelty, confidence_new)[0]
        txt = ax.text(0.85, 1.05, "p: {:.2f}".format(pc),
                      transform=ax.transAxes, size=10)

        ax = fig.add_subplot(grid[0, 1])
        x, y, c, order = quick_gaussian_kde(stddev, novelty)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_ylabel('log novelty')
        ax.set_ylim(0, np.percentile(novelty, 90))
        ax.set_xlabel('stddev')
        pc = stats.pearsonr(novelty, stddev)[0]
        txt = ax.text(0.85, 1.05, "p: {:.2f}".format(pc),
                      transform=ax.transAxes, size=10)

        ax = fig.add_subplot(grid[0, 2])
        x, y, c, order = quick_gaussian_kde(intensity, novelty)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_ylabel('log novelty')
        ax.set_ylim(0, np.percentile(novelty, 90))
        ax.set_xlabel('intensity')
        pc = stats.pearsonr(novelty, intensity)[0]
        txt = ax.text(0.85, 1.05, "p: {:.2f}".format(pc),
                      transform=ax.transAxes, size=10)

        ax = fig.add_subplot(grid[0, 3])
        x, y, c, order = quick_gaussian_kde(experr, novelty)
        ax.scatter(x[order], y[order], c=c[order], s=5, edgecolor='')
        ax.set_ylabel('log novelty')
        ax.set_ylim(0, np.percentile(novelty, 90))
        ax.set_xlabel('exp. error')
        pc = stats.pearsonr(novelty, experr)[0]
        txt = ax.text(0.85, 1.05, "p: {:.2f}".format(pc),
                      transform=ax.transAxes, size=10)

        filename = os.path.join(self.plot_path,
                                "%s_sign3_conf_scores.png" % sign3.dataset[:2])
        plt.savefig(filename, dpi=100)
        plt.close()

    def projection_gaussian_kde(self, x, y, limit=None, **kwargs):

        def quick_gaussian_kde(x, y, limit):
            xl = x[:limit]
            yl = y[:limit]
            xy = np.vstack([xl, yl])
            c = gaussian_kde(xy)(xy)
            order = c.argsort()
            return xl, yl, c, order

        if limit is None:
            limit = len(x)

        name = kwargs.get('name', 'PROJ')
        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        x, y, c, order = quick_gaussian_kde(x, y, limit)
        ax.scatter(x[order], y[order], c=c[order], **kwargs)
        ax.set_ylabel('')
        ax.set_xlabel('')
        filename = os.path.join(self.plot_path,
                                "2D_KDE_%s.png" % name)
        plt.savefig(filename, dpi=100)
        plt.close()

    def sign3_novelty_confidence(self, sign, limit=1000):

        fig = plt.figure(figsize=(3, 3))
        plt.subplots_adjust(left=0.2, right=1, bottom=0.2, top=1)
        gs = fig.add_gridspec(2, 2, wspace=0.0, hspace=0.0)
        gs.set_height_ratios((1, 5))
        gs.set_width_ratios((5, 1))

        ax_main = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
        ax_top.text(0.05, 0.2, "%s" % self.dataset[:2],
                    color=self.cc_colors(self.dataset),
                    transform=ax_top.transAxes,
                    name='Arial', size=14, weight='bold')
        ax_top.set_axis_off()
        ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
        ax_right.set_axis_off()

        # ax_main.plot((-1, 1), (-1, 1), ls="--",
        #             c="lightgray", alpha=.5)

        novelty = sign.get_h5_dataset('novelty_norm')[:limit]
        confidence = sign.get_h5_dataset('confidence')[:limit]
        sns.regplot(novelty, confidence,
                    ax=ax_main, n_boot=10000, truncate=False,
                    color=self.cc_colors(self.dataset),
                    scatter_kws=dict(s=10, edgecolor=''),
                    line_kws=dict(lw=1))

        pearson = stats.pearsonr(novelty, confidence)[0]
        ax_main.text(0.7, 0.9, r"$\rho$: {:.2f}".format(pearson),
                     transform=ax_main.transAxes,
                     name='Arial', size=10,
                     bbox=dict(facecolor='white', alpha=0.8, lw=0))

        ax_main.set_ylabel('')
        ax_main.set_xlabel('')
        ax_main.set_ylim(0.5, 1)
        ax_main.set_xlim(0, 1)
        ax_main.set_yticks([0.5, 1.0])
        ax_main.set_yticklabels(['0.5', '1'])
        ax_main.set_xticks([0, 0.5, 1.0])
        ax_main.set_xticklabels(['0', '0.5', '1'])
        ax_main.tick_params(labelsize=14, direction='inout')

        sns.distplot(novelty, ax=ax_top,
                     hist=False, kde_kws=dict(shade=True, bw=.2),
                     color=self.cc_colors(self.dataset))
        sns.distplot(confidence, ax=ax_right, vertical=True,
                     hist=False, kde_kws=dict(shade=True, bw=.2),
                     color=self.cc_colors(self.dataset))

        sns.despine(ax=ax_main, offset=3, trim=True)

        fig.text(0.5, -0.2, 'Novelty',
                 ha='center', va='center',
                 transform=ax_main.transAxes,
                 name='Arial', size=16)
        fig.text(-0.2, 0.5, 'Confidence', ha='center',
                 va='center', rotation='vertical',
                 transform=ax_main.transAxes,
                 name='Arial', size=16)

        # plt.tight_layout()
        outfile = os.path.join(
            self.plot_path, 'sign3_%s_novelty_confidence.png' % self.dataset_code)
        plt.savefig(outfile, dpi=100)
        plt.close('all')
