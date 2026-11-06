"""Chemical Checker entry-point.

The most common starting point in a CC project is the
:class:`ChemicalChecker`::

    from chemicalchecker import ChemicalChecker
    cc = ChemicalChecker()

When initializing a CC instance we usually want provide a root directory.
If, like in the example above, we don't specify anything, the default path is
assumed (that is the ``CC_ROOT`` variable in the
:class:`~chemicalchecker.util.config.Config`).

If the specified ``CC_ROOT`` directory is already populated, we have
successfully initialized the CC instance and will have access to its
signatures.

If the ``CC_ROOT`` directory is empty we proceed generating the CC directory
structure and we'll have an empty CC instance optimal for handling our own
signatures.

The organization of signatures under the ``CC_ROOT`` follows hierarchy of
``molset``/``dataset``/``signature``.

    * The ``molset`` is mostly for internal usage, and its expected values are
      either "full" or "reference". In some steps of the pipeline is
      convenient to work with the non-redundant set of signatures
      ("reference") while at end we want to map back to the "full" set of
      molecules.

    * The ``dataset`` is the bioactivity space of interest and is described by
      the _level_ (e.g. ``A``) the _sublevel_ (e.g. ``1``) and a _code_ for
      each input dataset starting from ``.001``. The directory structure
      follow this hierarchy (e.g. ``/root/full/A/A1/A1.001``)

    * The ``signature`` is one of the possible type of signatures (see
      :doc:`Signaturization <../signaturization>`) and the final path is
      something like ``/root/full/A/A1/A1.001/sign2``


Main goals of this class are:
    1. Check and enforce the directory structure behind a CC instance.
    2. Serve signatures to users or pipelines.
"""
import re
import os
import gzip
import h5py
import json
import shutil
import itertools
import wget
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from pathlib import Path

from .molkit import Mol
from .data import DataFactory
from .signature_data import DataSignature

from chemicalchecker.database import Dataset, Molecule
from chemicalchecker.database.database import test_connection
from chemicalchecker.util import logged, Config
from chemicalchecker.util.filesystem import FileSystem
from chemicalchecker.util.decorator import cached_property


@logged
class ChemicalChecker():
    """ChemicalChecker class."""

    def __init__(self, cc_root=None, custom_data_path=None, dbconnect=True):
        """Initialize a ChemicalChecker instance.

        If the CC_ROOT directory is empty a skeleton of CC is initialized.
        Otherwise the directory is explored and molset and datasets variables
        are discovered.

        Args:
            cc_root (None, str): The Chemical Checker root directory.
                If not specified the root is taken from the config file.
            custom_data_path (None, str): Path to one or more h5 files, detect
                their signature type, molset and dataset code form their
                'attrs' record.
            dbconnect (True, Bool): if True, try to connect to the DB
        """
        # Default cc_root is taken from config file
        if cc_root is None:
            self.cc_root = os.path.realpath(Config().PATH.CC_ROOT)
        else:
            self.cc_root = os.path.realpath(cc_root)

        self.db_access = dbconnect
        self._basic_molsets = ['reference', 'full']
        self._datasets = set()
        self._molsets = set(self._basic_molsets)
        self._exemplary_id = "001"
        self._metadata = None
        self.__log.debug("ChemicalChecker with root: %s", self.cc_root)
                
        # If non-existing CC_root
        if not os.path.isdir(self.cc_root):
            self.__log.debug("Empty root directory,"
                             "creating root and dataset dirs")
            original_umask = os.umask(0)
            os.makedirs(self.cc_root, 0o775)
            os.umask(original_umask)
            
            datasets = []
            # Test connection to DB, the database is not necessarily present
            self.__log.debug("Testing DB connection.")
            if dbconnect: 
                if test_connection():
                    for dataset in Dataset.get():
                        ds = dataset.dataset_code
                        datasets.append(ds)
                    
            else:
                for space in 'ABCDE':
                    for n in '12345':
                        datasets.append( space + n+'.'+self._exemplary_id )
                self.__log.debug("No database found, working locally.")
            
            for molset in self._basic_molsets:
                for ds in datasets:
                    new_dir = os.path.join(
                            self.cc_root, molset, ds[:1], ds[:2], ds)
                    self._datasets.add(ds)
                    original_umask = os.umask(0)
                    os.makedirs(new_dir, 0o775)
                    os.umask(original_umask)

        else:
            # if the directory exists get molsets and datasets
            paths = self._available_sign_paths(filename='sign*.h5')
            self._molsets = set(x.split('/')[-6] for x in paths)
            self._datasets = set(x.split('/')[-3] for x in paths)
            if custom_data_path is not None:
                self.__log.info("CC root directory exists: "
                                "ignoring 'custom_data_path'.")
                custom_data_path = None

        # import one or several custom signature files and models
        if custom_data_path is not None:
            if os.path.isfile(custom_data_path):
                raise Exception("'custom_data_path' must be a directory")

            custom_data_path = os.path.abspath(custom_data_path)
            self.__log.debug("Linking files from: %s" % custom_data_path)
            self.link(custom_data_path)

        self._molsets = sorted(list(self._molsets))
        self._datasets = [x for x in sorted(
            list(self._datasets)) if not x.endswith('000')]

    @property
    def coordinates(self):
        """Iterator on Chemical Checker coordinates."""
        for name, code in itertools.product("ABCDE", "12345"):
            yield name + code

    @property
    def datasets(self):
        """Iterator on Chemical Checker datasets."""
        for dataset in self._datasets:
            yield dataset

    @property
    def name(self):
        """Return the name of the Chemical Checker."""
        return os.path.basename(os.path.normpath(self.cc_root))

    @property
    def metadata(self):
        """Return the metadata of the Chemical Checker."""
        if self._metadata is None:
            self._metadata = self._get_metadata()
        return self._metadata

    def sign_metadata(self, key, molset, dataset, cctype):
        """Return the metadata of the Chemical Checker."""
        if self._metadata is None:
            self._metadata = self._get_metadata()

        if key not in self._metadata:
            raise Exception("Key `%s` not found in metadata" % key)
        if molset in self._metadata[key]:
            if dataset in self._metadata[key][molset]:
                if cctype in self._metadata[key][molset][dataset]:
                    return self._metadata[key][molset][dataset][cctype]
        try:
            sign = self.get_signature(cctype, molset, dataset)
        except Exception:
            self.__log.debug(
                "Signature not found, skipping: %s %s %s %s" %
                (self.name, *cctype, molset, dataset))
            return None
        if key == 'dimensions':
            return sign.shape
        if key == 'keys':
            return sign.keys

    def _get_metadata(self, overwrite=False):
        fn = os.path.join(self.cc_root, "metadata.json.zip")
        metadata = dict()
        if not os.path.exists(fn) or overwrite:
            metadata['dimensions'] = self.report_dimensions()
            metadata['keys'] = self.report_keys()
            json_str = json.dumps(metadata) + "\n"
            json_bytes = json_str.encode('utf-8')
            with gzip.open(fn, "w") as fh:
                fh.write(json_bytes)
            return metadata
        with gzip.open(fn, "r") as fh:
            json_bytes = fh.read()
        json_str = json_bytes.decode('utf-8')
        metadata = json.loads(json_str)
        return metadata

    def datasets_exemplary(self):
        """Iterator on Chemical Checker exemplary datasets."""
        for dataset in self.coordinates:
            yield dataset + '.001'

    @cached_property
    def universe(self):
        """Get the list of molecules in the CC universe.

        We define the CC universe as the union of all molecules found in sign0
        for any of the bioactivity datasets that are 'derived' and that are
        'essential'.
        """
        universe = set()
        dataset_accepted = []
        for ds in Dataset.get():
            if not ds.derived:
                self.__log.debug("Dataset '%s' not derived", ds)
                continue
            if not ds.essential:
                self.__log.debug("Dataset '%s' not essential", ds)
                continue
            s0 = self.get_signature('sign0', 'full', ds.code)
            dataset_accepted.append(ds.code)
            try:
                universe.update(s0.unique_keys)
            except Exception as ex:
                self.__log.warning(str(ex))
        self.__log.debug("Datasets defining universe: %s",
                         ' '.join(dataset_accepted))
        return sorted(list(universe))

    def get_universe_inchi(self):
        self.__log.debug("Fetching InChI of universe")
        ink_inchi = Molecule.get_inchikey_inchi_mapping(self.universe)
        return [ink_inchi[k] for k in self.universe]

    @staticmethod
    def set_verbosity(level='warning', logger_name='chemicalchecker',
                      format=None):
        '''Set the verbosity for logging module.'''
        import logging
        level = level.upper()
        levels = {'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}
        log_fn = {'DEBUG': ChemicalChecker.__log.debug,
                  'INFO': ChemicalChecker.__log.info,
                  'WARNING': ChemicalChecker.__log.warning,
                  'ERROR': ChemicalChecker.__log.error,
                  'CRITICAL': ChemicalChecker.__log.critical}
        logger = logging.getLogger(logger_name)
        if level == 'DEBUG':
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s %(name)-12s [%(levelname)-8s] %(message)s')
            ch.setFormatter(formatter)
            logger.handlers = []
            logger.addHandler(ch)
        else:
            ch = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(levelname)-8s] %(message)s')
            ch.setFormatter(formatter)
            logger.handlers = []
            logger.addHandler(ch)
        logger.setLevel(levels[level])
        log_fn[level]("Logging level %s for logger '%s'." %
                      (level.upper(), logger_name))

    def _available_sign_paths(self, molset='[!exemplary]*', dataset='*', signature='*',
                              filename='*.h5'):
        paths = glob(os.path.join(self.cc_root, molset, '*',
                                  '*', dataset, signature, filename))
        return paths

    def report_available(self, molset='*', dataset='*', signature='*'):
        """Report available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Use arguments to apply filters.

        Args:
            molset (str, optional): Filter for the moleculeset e.g. 'full' or
                'reference'
            dataset (str, optional): Filter for the dataset e.g. A1.001
            signature (str, optional): Filter for signature type e.g. 'sign1'

        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = self._available_sign_paths(molset, dataset, signature)
        molset_dataset_sign = dict()
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            if molset not in molset_dataset_sign:
                molset_dataset_sign[molset] = dict()
            if dataset not in molset_dataset_sign[molset]:
                molset_dataset_sign[molset][dataset] = list()
            molset_dataset_sign[molset][dataset].append(sign)
            molset_dataset_sign[molset][dataset].sort()
        return molset_dataset_sign

    def report_keys(self, molset='full', dataset='*', signature='sign1'):
        """Report keys of all available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Report the list of keys. Use arguments to apply filters.
        Args:
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = self._available_sign_paths(molset, dataset, signature)
        molset_dataset_sign = dict()
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            if molset not in molset_dataset_sign:
                molset_dataset_sign[molset] = dict()
            if dataset not in molset_dataset_sign[molset]:
                molset_dataset_sign[molset][dataset] = dict()
            try:
                with h5py.File(path, 'r') as fh:
                    if 'keys' not in fh.keys():
                        continue
                    decoder = np.vectorize(lambda x: x.decode())
                    molset_dataset_sign[molset][
                        dataset][sign] = decoder(fh['keys'][:]).tolist()
            except Exception as ex:
                self.__log.warning(
                    'problem reading file %s: %s' % (path, str(ex)))
        return molset_dataset_sign

    def download_all_sign_links(self, out_directory):
        if( not os.path.isdir(out_directory) ):
            os.mkdir(out_directory)
        wget.download( "https://chemicalchecker.com/api/db/getFile/root/sign_links.tar.gz", out=out_directory )
    
    def download_all_ftp_signatures(self, out_directory, type_sign='sign2', only_chemical = False):
        if( not os.path.isdir(out_directory) ):
            os.mkdir(out_directory)
            
        if( os.path.isdir(out_directory) ):
            if( type_sign in ['sign0', 'sign1', 'sign2', 'sign3'] ):
                prefix = f"signature{ type_sign[-1] }"
                spaces = ['A','B', 'C', 'D', 'E']
                if( only_chemical ):
                    spaces = ['A']
                    
                combinations = []
                for s in spaces:
                    for i in range(1, 6):
                        combinations.append( s+str(i) )
                
                for c in tqdm(combinations):
                    if( type_sign == 'sign3' ):
                        wget.download( f"https://chemicalchecker.com/api/db/getFile/root/{ c }.h5", out=out_directory )
                        os.system( f"mv {out_directory}/{ c }.h5 {out_directory}/{ c }_sign3.h5" )
                    else:
                        wget.download( f"https://chemicalchecker.com/api/db/getFile/{prefix}/{ c }_{ type_sign }.h5", out=out_directory )
            else:
                self.__log.warning(
                        'Invalid signature option: %s. This function is compatible with sign0, sign1, sign2 and sign3' % ( type_sign ) )
        else:
            self.__log.warning(
                        'Output directory does not exist: %s' % ( out_directory ) )

    def download_all_ftp_signatures(self, out_directory, type_sign='sign2', only_chemical = False):
        if( not os.path.isdir(out_directory) ):
            os.mkdir(out_directory)
            
        if( os.path.isdir(out_directory) ):
            if( type_sign in ['sign0', 'sign1', 'sign2', 'sign3'] ):
                prefix = f"signature{ type_sign[-1] }"
                spaces = ['A','B', 'C', 'D', 'E']
                if( only_chemical ):
                    spaces = ['A']
                    
                combinations = []
                for s in spaces:
                    for i in range(1, 6):
                        combinations.append( s+str(i) )
                
                for c in tqdm(combinations):
                    if( type_sign == 'sign3' ):
                        wget.download( f"https://chemicalchecker.com/api/db/getFile/root/{ c }.h5", out=out_directory )
                        os.system( f"mv {out_directory}/{ c }.h5 {out_directory}/{ c }_sign3.h5" )
                    else:
                        wget.download( f"https://chemicalchecker.com/api/db/getFile/{prefix}/{ c }_{ type_sign }.h5", out=out_directory )
            else:
                self.__log.warning(
                        'Invalid signature option: %s. This function is compatible with sign0, sign1, sign2 and sign3' % ( type_sign ) )
        else:
            self.__log.warning(
                        'Output directory does not exist: %s' % ( out_directory ) )
    
    def copy_ftp_signatures_in_cc_root(self, local_cc_dir, sign_directory, type_sign='sign2'):
        if( os.path.isdir(local_cc_dir) ):
            if( os.path.isdir(sign_directory) ):
                if( type_sign in ['sign0', 'sign1', 'sign2', 'sign3'] ):
                    spaces = ['A','B', 'C', 'D', 'E']
                    combinations = []
                    for s in spaces:
                        for i in range(1, 6):
                            combinations.append( s+str(i) )
                    
                    for c in tqdm(combinations):
                        source = os.path.join( sign_directory, f"{ c }_{ type_sign }.h5" )
                            
                        sign_folder = os.path.join( local_cc_dir, "full", c[0], c, f"{c}.001", type_sign )
                        if( not os.path.isdir(sign_folder) ):
                            os.mkdir( sign_folder )
                        dest = os.path.join( local_cc_dir, "full", c[0], c, f"{c}.001", type_sign, f"{ type_sign }.h5" )
                        if( not os.path.isfile( dest ) ):
                            os.symlink( source, dest )
                else:
                    self.__log.warning(
                            'Invalid signature option: %s. This function is compatible with sign0, sign1, sign2 and sign3' % ( type_sign ) )
            else:
                self.__log.warning(
                            'Signatures download directory does not exist: %s' % ( sign_directory ) )
        else:
            self.__log.warning(
                            'CC local directory does not exist: %s' % ( local_cc_dir ) )
    
    def report_dimensions(self, molset='*', dataset='*', signature='*',
                          matrix='V'):
        """Report dimensions of all available signatures in the CC.

        Get the moleculeset/dataset combination where signatures are available.
        Report the size of the 'V' matrix. Use arguments to apply filters.
        Args:
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = self._available_sign_paths(molset, dataset, signature)
        molset_dataset_sign = dict()
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            if molset not in molset_dataset_sign:
                molset_dataset_sign[molset] = dict()
            if dataset not in molset_dataset_sign[molset]:
                molset_dataset_sign[molset][dataset] = dict()
            try:
                with h5py.File(path, 'r') as fh:
                    if matrix not in fh.keys():
                        continue
                    molset_dataset_sign[molset][
                        dataset][sign] = fh[matrix].shape
            except Exception as ex:
                self.__log.warning(
                    'problem reading file %s: %s' % (path, str(ex)))
        return molset_dataset_sign

    def report_status(self, molset='*', dataset='*', signature='*'):
        """Report status of signatures in the CC.

        Args:
            molset(str): Filter for the moleculeset e.g. 'full' or 'reference'
            dataset(str) Filter for the dataset e.g. A1.001
            signature(str): Filter for signature type e.g. 'sign1'
        Returns:
            Nested dictionary with molset, dataset and list of signatures
        """
        paths = self._available_sign_paths(molset, dataset, signature)
        molset_dataset_sign = dict()
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            if molset not in molset_dataset_sign:
                molset_dataset_sign[molset] = dict()
            if dataset not in molset_dataset_sign[molset]:
                molset_dataset_sign[molset][dataset] = dict()
            try:
                status_file = os.path.join(os.path.dirname(path), '.STATUS')
                if not os.path.isfile(status_file):
                    status = ('N/A', 'STATUS file not found!')
                else:
                    with open(status_file, 'r') as fh:
                        for line in fh.readlines():
                            status = line.strip().split('\t')
                molset_dataset_sign[molset][dataset][sign] = status
            except Exception as ex:
                self.__log.warning(
                    'problem reading file %s: %s' % (path, str(ex)))
        return molset_dataset_sign

    def get_signature_path(self, cctype, molset, dataset_code):
        """Return the signature path for the given dataset code.

        This should be the only place where we define the directory structure.
        The signature directory tipically contain the signature HDF5 file.

        Args:
            cctype(str): The Chemical Checker datatype i.e. one of the sign*.
            molset(str): The molecule set name.
            dataset_code(str): The dataset of the Chemical Checker.
        Returns:
            signature_path(str): The signature path.
        """
        signature_path = os.path.join(self.cc_root, molset, dataset_code[:1],
                                      dataset_code[:2], dataset_code, cctype)

        # self.__log.debug("signature path: %s", signature_path)
        return signature_path

    def get_signature(self, cctype, molset, dataset_code, *args, **kwargs):
        """Return the signature for the given dataset code.

        Args:
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*).
            molset(str): The molecule set name.
            dataset_code(str): The dataset code of the Chemical Checker.
            params(dict): Optional. The set of parameters to initialize and
                compute the signature. If the signature is already initialized
                this argument will be ignored.
            as_dataframe(bool): True to get the signature as pandas DataFrame.
        Returns:
            data(Signature): A `Signature` object, the specific type depends
                on the cctype passed.
        """
        signature_path = self.get_signature_path(cctype, molset, dataset_code)
        as_dataframe = kwargs.pop('as_dataframe', False)
        # the factory will return the signature with the right class
        data = DataFactory.make_data(
            cctype, signature_path, dataset_code, *args, **kwargs)
        if as_dataframe:
            return data.as_dataframe()
        return data

    def get_data_signature(self, cctype, dataset_code):
        """Return the data signature for the given dataset code.

        Args:
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*).
            dataset_code(str): The dataset code of the Chemical Checker.
        Returns:
            data(Signature): A `DataSignature` object, the specific type
                depends on the cctype passed.
                It only allows access to the sign data.
        """
        args = ()
        kwargs = {}
        molset = "full"
        if len(dataset_code) == 2:
            dataset_code = dataset_code + "." + self._exemplary_id
        signature_path = self.get_signature_path(cctype, molset, dataset_code)
        # the factory will return the signature with the right class
        data = DataFactory.make_data(
            cctype, signature_path, dataset_code, *args, **kwargs)
        if not os.path.exists(data.data_path):
            self.__log.error(
                "There is no data for %s and dataset code %s" %
                (cctype, dataset_code))
            return None
        return DataSignature(data.data_path)

    def signature(self, dataset, cctype, as_dataframe=False, **kwargs):
        return self.get_signature(cctype=cctype, molset="full",
                                  dataset_code=dataset,
                                  as_dataframe=as_dataframe, **kwargs)

    @staticmethod
    def get_h5_metadata(fn, format_dict):
        """ Return H5 metadata.

        Returns:
            tuple of the type ('full', 'A', 'A1', 'A1.001','sign3', h5_path) 
            or None if something's wrong
        """
        out = []
        with h5py.File(fn, 'r') as ccfile:
            # check if the required info is presents in the h5 file
            for requiredKey, requiredFormat in format_dict.items():
                if requiredKey not in ccfile.attrs:
                    ChemicalChecker.__log.warning(
                        "Attribute {} cannot be retrieved from {},"
                        " skipping this file".format(
                            requiredKey, ccfile))
                    return None
                else:
                    # check the format of the provided info
                    matching = requiredFormat.match(
                        ccfile.attrs[requiredKey])
                    if matching is None:
                        ChemicalChecker.__log.warning(
                            "Problem with format",
                            ccfile.attrs[requiredKey])
                        return None
            # Prepare the signature file name and path
            out.append(ccfile.attrs['molset'].lower())
            out.append(ccfile.attrs['dataset_code'][0])
            out.append(ccfile.attrs['dataset_code'][:2])
            out.append(ccfile.attrs['dataset_code'])
            out.append(ccfile.attrs['cctype'].lower())
            out.append(fn)
        return tuple(out)

    @staticmethod
    def get_model_metadata(fn, format_dict):
        """ Return model metadata.

        Returns:
            tuple of the type ('full', 'A', 'A1', 'A1.001','sign3', h5_path) 
            or None if something's wrong
        """
        out = []
        mdl_meta = json.load(open(os.path.join(fn, 'metadata.json'), 'r'))
        # check if the required info is presents in the model file
        for requiredKey, requiredFormat in format_dict.items():
            if requiredKey not in mdl_meta.keys():
                ChemicalChecker.__log.warning(
                    "Attribute {} cannot be retrieved from {},"
                    " skipping this file".format(
                        requiredKey, mdl_meta))
                return None
            else:
                # check the format of the provided info
                matching = requiredFormat.match(
                    mdl_meta[requiredKey])
                if matching is None:
                    ChemicalChecker.__log.warning(
                        "Problem with format",
                        mdl_meta[requiredKey])
                    return None
        # Prepare the model file name and path
        out.append(mdl_meta['molset'].lower())
        out.append(mdl_meta['dataset_code'][0])
        out.append(mdl_meta['dataset_code'][:2])
        out.append(mdl_meta['dataset_code'])
        out.append(mdl_meta['cctype'].lower())
        out.append(fn)
        return tuple(out)

    @staticmethod
    def get_metadatas(files, metadata_func, format_dict):
        """Extract the metadata from files using a function."""
        available_files = ", ".join([os.path.basename(f) for f in files])
        ChemicalChecker.__log.debug(
            "Importing files {}".format(available_files))
        # Keep files containing the required info in the correct format
        metadatas = list()
        for file in files:
            res = metadata_func(file, format_dict)
            if res:
                metadatas.append(res)
            else:  # guess from filename
                name = Path(file).stem
                molset = 'full'
                if( len(name.split('_'))==3):
                    cctype, dataset_code, molset = name.split('_')
                else:
                    dataset_code, cctype = name.split('_')
                    dataset_code = dataset_code+'.001'
                    
                res = list()
                res.append(molset.lower())
                res.append(dataset_code[0])
                res.append(dataset_code[:2])
                res.append(dataset_code)
                res.append(cctype.lower())
                res.append(file)
                metadatas.append(res)
        if len(metadatas) == 0:
            raise Exception(
                "None of the provided h5 datasets have sufficient info in"
                " its attributes! Please ensure myh5file.attrs has the"
                "folllowing keys: 'dataset_code', 'cctype', 'molset'")
        return metadatas

    def link(self, custom_data_path):
        """Link H5 files and models from a given custom directory.

        Populates local CC instance with symlinks to external signatures H5s
        or models.

        Args:
            custom_data_path(str): Path to a directory signature containing H5s
                models or symlinks.
        """
        # define required format for metadata fields
        ds_fmt = re.compile(r"[A-Z]\d\.\d\d\d")  # dataset code (ex: A1.001)
        cctype_fmt = re.compile(r"sign\d")
        molset_fmt = re.compile(r"(full|reference)", re.IGNORECASE)
        format_dict = dict(dataset_code=ds_fmt,
                           cctype=cctype_fmt, molset=molset_fmt)

        metapath = os.path.join(custom_data_path, 'metadata.json.zip')
        metadest = os.path.join(self.cc_root, 'metadata.json.zip')
        if( os.path.isfile( metapath) ):
            shutil.copyfile( metapath, metadest)
            
        # get H5 files metadata
        files = glob(os.path.join(custom_data_path, "*.h5"))
        if len(files) == 0:
            ChemicalChecker.__log.warning(f"No *.h5 file found in {custom_data_path}, "
                                          "CC instance will be without data.")
        else:
            metadatas = self.get_metadatas(
                files, self.get_h5_metadata, format_dict)

            # Create the CC instance folder structure
            for meta in metadatas:
                sign_path = os.path.join(self.cc_root, *meta[:-1])
                # create dir
                os.makedirs(sign_path, exist_ok=True)
                src = meta[-1]
                # symbolic link to the h5 file in the cc_repo as signx.h5
                dst = os.path.join(sign_path, meta[-2] + '.h5')
                self.__log.debug("%s ==> %s" % (src, dst))
                os.symlink(src, dst)

        # get models metadata
        files = glob(os.path.join(custom_data_path, "*.models"))
        if len(files) == 0:
            ChemicalChecker.__log.warning(f"No *.model found in {custom_data_path}, "
                                          "CC instance will be without models.")
        else:
            metadatas = self.get_metadatas(
                files, self.get_model_metadata, format_dict)

            # add symlinks to models
            for meta in metadatas:
                sign_path = os.path.join(self.cc_root, *meta[:-1])
                # create dir
                os.makedirs(sign_path, exist_ok=True)
                src = meta[-1]
                # symbolic link to the models path in the cc_repo as models folder
                dst = os.path.join(sign_path, 'models')
                self.__log.debug("%s ==> %s" % (src, dst))
                os.symlink(src, dst)

    def export(self, destination, signature, h5_filter=None,
               h5_names_map={}, overwrite=False, version=None):
        """Export a signature h5 file to a given path.

        Which dataset to copy can be specified as well as how to rename some
        dataset.

        Args:
            destination(str): A destination path.
            signature(sign): A signature object.
            h5_filter(list): List of h5 dataset name to export.
            h5_names_map(dict): Dictionary of current to final h5 dataset name.
            overwrite(boo): Whether to allow overwriting the export.
            version(int): Mark the exported signature with a version number.
        """
        src_file = signature.data_path

        if not os.path.isfile(src_file):
            raise Exception('Signature must have an H5 file to export!')
        if not overwrite and os.path.isfile(destination):
            raise Exception('File %s already exists!' % destination)

        src = h5py.File(signature.data_path, 'r')
        dst = h5py.File(destination, 'w')
        if h5_filter is None:
            h5_filter = src.keys()
        h5_names_map = {x: x for x in h5_filter}
        h5_names_map.update(h5_names_map)
        for h5ds in h5_filter:
            if h5ds not in src:
                raise Exception('%s not available in %s' % (h5ds, src_file))
            h5src = src[h5ds][:]
            dst.create_dataset(h5_names_map[h5ds], data=h5src)
        if version is not None:
            dst.create_dataset('version', data=[version])

        # adding metadata
        attributes = dict(dataset_code=signature.dataset,
                          cctype=signature.cctype, molset=signature.molset)
        if len(dst.attrs) != 3:
            for k, v in attributes.items():
                dst.attrs[k] = v

        src.close()
        dst.close()

    # Export minimum CC zipped folder
    def export_cc(self, root_destination, folder_destination):
        """Export a zipped folder containing the minimum files necessary 
           to run a complete CC protocol

        It includes:
            - full: all sign0 (.h5 files + fit.ready file in models folder)
            - reference: sign1 models folder (.pkl files only) 
              --> sign1 are going to be generated based on sign0
                  at the initialization of the ChemicalChecker instance
            - reference: sign2 models (savedmodel folder only) 
              --> sign2 are going to be generated based on sign1 and neig1
              (also generated once sign1 is ready)

        Args:
            root_destination(str): An export destination path
            folder_destination(str): additional path to append to root_destination
                --> used to define the base_dir when zipping
        """
        destination = os.path.join(root_destination, folder_destination)
        self.__log.debug('Exporting CC {} to {}'.format(
            self.cc_root, destination))

        for molset in self._basic_molsets:
            self.__log.debug('Molset: {}'.format(molset))
            for dataset in self.datasets_exemplary():
                self.__log.debug('Dataset: {}'.format(dataset))
                # 0) root dir of a dataset: full/reference
                # adding an additional layer inside the folder_destination so that when the
                # folder is zipped, it remains inside the CC version directory at the same level as
                # the exported sign3
                new_dir = os.path.join(
                    destination, folder_destination, molset, dataset[:1], dataset[:2], dataset)
                FileSystem.check_dir_existance_create(new_dir)
                # 1) sign0
                if molset == 'full':
                    sign_type = 'sign0'
                    sign = self.get_signature(sign_type, molset, dataset)
                    sign_dir = os.path.join(new_dir, '%s' % sign_type)
                    FileSystem.check_dir_existance_create(sign_dir)
                    FileSystem.check_dir_existance_create(sign_dir, ['models'])
                    dst = os.path.join(sign_dir, '%s.h5' % sign_type)
                    if not os.path.isfile(dst):
                        self.__log.debug(
                            "Copying {} to {}".format(sign, sign_dir))
                        self.export(dst, sign)
                    fit_file = os.path.join(sign_dir, 'models', 'fit.ready')
                    FileSystem.check_file_existance_create(fit_file)
                # 2) sign1
                if molset == 'reference':
                    sign_type = 'sign1'
                    sign = self.get_signature(sign_type, molset, dataset)
                    sign_dir = os.path.join(new_dir, '%s' % sign_type)
                    dst_models_path = FileSystem.check_dir_existance_create(sign_dir, [
                                                                            'models'])
                    regex = re.compile(
                        r'.*\.pkl$'
                        r'|.*\.pkl\.projection(\..*\.npy)?$'
                        r'|.*\.joblib$')
                    self.__log.debug("Exporting files of {} from \
                                        {} to {}".format(sign_type, sign.model_path, dst_models_path))
                    for _, _, src_files in os.walk(sign.model_path):
                        for src_file in src_files:
                            if regex.match(src_file):
                                dst_file = os.path.join(
                                    dst_models_path, '%s' % src_file)
                                FileSystem.check_file_existance_create(
                                    dst_file)
                                shutil.copyfile(os.path.join(
                                    sign.model_path, src_file), dst_file)
                # 3) sign2
                if molset == 'reference':
                    sign_type = 'sign2'
                    model = 'adanet'
                    sign = self.get_signature(sign_type, molset, dataset)
                    sign_dir = os.path.join(new_dir, '%s' % sign_type)
                    dst_models_adanet_path = FileSystem.check_dir_existance_create(sign_dir, [
                                                                                   'models', model])
                    if os.path.isdir(os.path.join(dst_models_adanet_path, 'savedmodel')):
                        self.__log.debug("savedmodel folder already exists in {} models directory\n \
                                Removing it".format(sign_type))
                        shutil.rmtree(os.path.join(
                            dst_models_adanet_path, 'savedmodel'))
                    src_path = os.path.join(
                        sign.model_path, model, 'savedmodel')
                    self.__log.debug("Copying savedmodel folder from {} to {}".format(
                        os.path.join(sign.model_path, model), dst_models_adanet_path))
                    shutil.copytree(src_path, os.path.join(
                        dst_models_adanet_path, 'savedmodel'))

        # zipping the destination folder
        complete_destination = os.path.join(destination, folder_destination)
        self.__log.debug(
            'Zipping exported CC folder {}'.format(complete_destination))
        shutil.make_archive(complete_destination, 'gztar',
                            destination, folder_destination)
        if os.path.isfile(os.path.join(destination, folder_destination + '.tar.gz')):
            shutil.rmtree(complete_destination)

    def check_dir_existance_create(dir_path, additional_path=None):
        """Args:
            dir_path(str): root path
            additional_path(list) : list of strings including additional
                                    path parts to append to the root path
        """
        path = dir_path
        if additional_path:
            for element in additional_path:
                path = os.path.join(path, element)
        if not os.path.isdir(path):
            original_umask = os.umask(0)
            os.makedirs(path, 0o775)
            os.umask(original_umask)
        return path

    def check_file_existance_create(file_path):
        """
            This method create an empty file if it doesn't exist already
        """
        if not os.path.isfile(file_path):
            with open(file_path, 'w'):
                pass

    def symlink_to(self, source_cc, cctypes=['sign0'],
                   molsets=['reference', 'full'], datasets='exemplary',
                   rename_dataset=None, models=False):
        """Link current CC instance to other via symlinks.

        When experimenting with signature parameters it's useful to have
        low cctype (e.g. sign0, sign1) not copied but simply linked.

        Args:
            source_cc(ChemicalChecker): A different CC instance to link.
            cctypes(list): The signature (i.e. sign*) to link.
            molsets(list): The molecule set name to link.
            datasets(list): The codes of dataset to link.
            rename_dataset(dict): None by default which to no renaming.
                Otherwise a mapping of source to destination name should be
                provided.
            models(bool): If True, models directory will also be linked.
                This will delete the local models for the specified
                datasets.
        """
        if datasets == 'exemplary':
            datasets = list(self.datasets_exemplary())

        if rename_dataset is None:
            rename_dataset = dict(zip(datasets, datasets))
        for molset in molsets:
            for ds in datasets:
                dst_ds = rename_dataset[ds]
                dst_ds_dir = os.path.join(
                    self.cc_root, molset, dst_ds[:1], dst_ds[:2], dst_ds)
                os.makedirs(dst_ds_dir, exist_ok=True)
                src_ds_dir = os.path.join(
                    source_cc.cc_root, molset, ds[:1], ds[:2], ds)
                for cctype in cctypes:
                    dst_dir = os.path.join(dst_ds_dir, cctype)
                    src_dir = os.path.join(src_ds_dir, cctype)
                    self.__log.debug("Link %s --> %s", dst_dir, src_dir)
                    # if the destination already a folder then the user has
                    # full ownership over it, otherwise just symlink the ref.
                    if os.path.isdir(dst_dir):
                        self.__log.warning("%s already present", dst_dir)
                        # in case the directory already exists try to link
                        # the reference model dir
                        if models:
                            dst_model_dir = os.path.join(dst_dir, 'models')
                            if os.path.isdir(dst_model_dir):
                                if os.path.islink(dst_model_dir):
                                    os.remove(dst_model_dir)
                                else:
                                    shutil.rmtree(dst_model_dir)
                            src_model_dir = os.path.join(src_dir, 'models')
                            os.symlink(src_model_dir, dst_model_dir)
                        continue
                    else:
                        os.symlink(src_dir, dst_dir)

    def copy_signature_from(self, source_cc, cctype, molset, dataset_code,
                            overwrite=False):
        """Copy a signature file from another CC instance.

        Args:
            source_cc(ChemicalChecker): A different CC instance.
            cctype(str): The Chemical Checker datatype (i.e. one of the sign*).
            molset(str): The molecule set name.
            dataset_code(str): The dataset code of the Chemical Checker.
        """
        # initialize destination
        dst_signature_path = self.get_signature_path(
            cctype, molset, dataset_code)
        dst_sign = DataFactory.make_data(
            cctype, dst_signature_path, dataset_code)
        # initializa source
        src_signature_path = source_cc.get_signature_path(
            cctype, molset, dataset_code)
        src_sign = DataFactory.make_data(
            cctype, src_signature_path, dataset_code)
        # copy data
        src = src_sign.data_path
        dst = dst_sign.data_path
        self.__log.info("Copying signature from %s to %s", src, dst)
        if not os.path.isfile(src):
            raise Exception("Source file %s does not exists.", src)
        if os.path.isfile(dst):
            self.__log.info("File %s exists already.", dst)
            if not overwrite:
                raise Exception("File %s exists already.", dst)
        shutil.copyfile(src, dst)

    def _assert_equal(self, other_cc, cctypes=['sign1', 'sign2', 'sign3'],
                      molsets=['full', 'reference']):
        """Compare two ChemicalChecker instances."""
        for ds in self.datasets_exemplary():
            for cctype in ['sign1', 'sign2']:
                for molset in ['full', 'reference']:
                    s1 = self.get_signature(cctype, molset, ds)
                    s2 = other_cc.get_signature(cctype, molset, ds)
                    assert(all(s1[0] == s2[0]))
                    assert(all(s1[-1] == s2[-1]))
                    assert(s1.info_h5 == s2.info_h5)
        return True

    def get_molecule(self, mol_str, str_type=None):
        """Return a molecule `Mol` object.

        Args:
            mol_str: Compound identifier (e.g. SMILES string)
            str_type: Type of identifier ('inchikey', 'inchi' and 'smiles' are
                accepted) if 'None' we do our best to guess.
        """
        return Mol(self, mol_str, str_type=str_type)

    def get_global_signature(self, mol_str, str_type=None):
        """Return the (stacked) global signature
        If the given molecule belongs to the universe.


        Args:
            mol_str: Compound identifier (e.g. SMILES string)
            str_type: Type of identifier ('inchikey', 'inchi' and 'smiles' are
                accepted) if 'None' we do our best to guess.
        """
        try:
            mol = self.get_molecule(mol_str, str_type)
        except Exception as e:
            self.__log.warning(
                "Problem with generating molecule object from " + mol_str)
            self.__log.warning(e)
            return None

        if mol.inchikey in set(self.universe):
            spaces = [''.join(t) for t in itertools.product(
                'ABCDE', '12345', ['.001'])]
            try:
                global_sign = np.concatenate(
                    [mol.signature('sign3', sp) for sp in spaces], axis=0)
            except Exception as e2:
                self.__log.warning(
                    "Problem with generating global signature from " + mol_str)
                self.__log.warning(e2)
            else:
                return global_sign

        else:
            self.__log.warning(mol_str + " NOT IN UNIVERSE")

        return None

    def export_symlinks(self, dest_path=None, signatures=True, models=True):
        """Creates symlinks for all available signatures H5 or models path
           in a single folder.

        Args:
            dest_path (str): The destination for symlink, if None then the
                default under the cc_root is generated.
            signatures (bool): export signature files.
            models (bool): export models paths.
        """
        if dest_path is None:
            dest_path = os.path.join(self.cc_root, 'sign_links')

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        molsets = ['full', 'reference']
        cctypes = ['sign0', 'sign1', 'sign2', 'sign3', 'sign4']
        ds = self.coordinates
        for molset, cctype, ds in itertools.product(molsets, cctypes, ds):
            dataset_code = ds + '.001'
            sign = self.get_signature(cctype, molset, dataset_code)
            sign_file = sign.data_path

            if signatures and os.path.isfile(sign_file):
                dst_name = "_".join([cctype, dataset_code, molset])
                dst_name += ".h5"
                # Make a symlink into the destination
                symlink = os.path.join(dest_path, dst_name)
                try:
                    os.symlink(sign_file, symlink)
                except Exception as ex:
                    self.__log.warning(
                        "Error creating %s: %s" % (symlink, str(ex)))

            model_path = os.path.join(sign.signature_path, 'models')
            if models and os.path.isdir(model_path):
                dst_name = "_".join([cctype, dataset_code, molset])
                dst_name += ".models"
                # Make a symlink into the destination
                symlink = os.path.join(dest_path, dst_name)
                try:
                    os.symlink(model_path, symlink)
                except Exception as ex:
                    self.__log.warning(
                        "Error creating %s: %s" % (symlink, str(ex)))

    @staticmethod
    def _check_h5_copy(src, dst, metadata):
        """Raise unless `dst` is `src` plus exactly `metadata`."""
        with h5py.File(src, 'r') as a, h5py.File(dst, 'r') as b:
            for k, v in metadata.items():
                if b.attrs.get(k) != v:
                    raise Exception('attribute %s not written' % k)
            if set(a.keys()) != set(b.keys()):
                raise Exception('dataset names differ')
            for k in a.keys():
                if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype:
                    raise Exception('%s shape/dtype differ' % k)
                # `V` can be 34 GB, so the copy is not re-read in full: both
                # ends are enough to catch a truncated or shifted write
                for sl in (slice(0, 1000), slice(-1000, None)):
                    if not np.array_equal(a[k][sl], b[k][sl]):
                        raise Exception('%s data differ' % k)

    def _stamp_h5_metadata(self, path, metadata):
        """Write root attributes to an H5 file without ever mutating it.

        The attributes are a few hundred bytes, but HDF5 has no journal: a
        crash while the root object header is rewritten can leave the file
        unopenable, and these signatures are weeks of compute with no backup.
        So stamp a copy, check the copy is intact, and swap it in with
        `os.replace`, which is atomic within a directory. An interrupted run
        leaves the original untouched and a `.stamping` file to delete.
        """
        tmp = path + '.stamping'
        try:
            shutil.copy2(path, tmp)
            with h5py.File(tmp, 'a') as fh:
                for k, v in metadata.items():
                    if k in fh.attrs:
                        del fh.attrs[k]
                    fh.attrs.create(name=k, data=v)
            self._check_h5_copy(path, tmp, metadata)
            os.replace(tmp, path)
        except Exception:
            if os.path.isfile(tmp):
                os.remove(tmp)
            raise

    def add_sign_metadata(self, molset='*', dataset='*', signature='*'):
        """Add metadata to available signatures.

        Signature H5 files carry `cctype`, `dataset_code` and `molset` as root
        attributes. `ChemicalChecker(root, custom_data_path=...)` rebuilds the
        directory tree from them, so a downloaded flat folder of signatures is
        only usable if they are set.

        Files already carrying the right metadata are skipped, which makes
        re-running free. The rest are stamped via `_stamp_h5_metadata`, which
        never writes into the original.

        Args:
            molset (str, optional): Filter for the moleculeset e.g. 'full' or
                'reference'
            dataset (str, optional): Filter for the dataset e.g. A1.001
            signature (str, optional): Filter for signature type e.g. 'sign1'
        """
        paths = self._available_sign_paths(molset, dataset, signature)
        for path in paths:
            metadata = dict(cctype=path.split('/')[-2],
                            dataset_code=path.split('/')[-3],
                            molset=path.split('/')[-6])
            try:
                with h5py.File(path, 'r') as fh:
                    current = dict(fh.attrs)
                # compared by value, not just presence: a wrongly stamped file
                # is the case that most needs fixing
                if all(current.get(k) == v for k, v in metadata.items()):
                    continue
                self.__log.debug("Stamping metadata on: %s" % path)
                self._stamp_h5_metadata(path, metadata)
            except Exception as ex:
                self.__log.warning(
                    "Could not add metadata to %s: %s" % (path, ex))
                continue

    def add_model_metadata(self, molsets=['full', 'reference'], dataset='*',
                           signature='sign*'):
        """Add metadata to available models.

        Args:
            molset (str, optional): Filter for the moleculeset e.g. 'full' or
                'reference'
            dataset (str, optional): Filter for the dataset e.g. A1.001
            signature (str, optional): Filter for signature type e.g. 'sign1'
        """
        paths = list()
        for molset in molsets:
            paths.extend(self._available_sign_paths(molset, dataset, signature,
                                                    'models'))
        for path in paths:
            molset = path.split('/')[-6]
            dataset = path.split('/')[-3]
            sign = path.split('/')[-2]
            metadata = dict(cctype=sign, dataset_code=dataset, molset=molset)
            metadata_file = os.path.join(path, 'metadata.json')
            try:
                with open(metadata_file, 'w') as fh:
                    json.dump(metadata, fh)
            except Exception:
                self.__log.warning("Could not add metadata to: %s" % path)
                continue
