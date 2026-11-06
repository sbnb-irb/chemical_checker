import os
import math
import h5py
import json
import pickle
import shutil
import tempfile
import collections

import numpy as np
from tqdm import tqdm

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.database import Dataset

from chemicalchecker.util import psql
from chemicalchecker.util import Config
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged, HPC


# We got these strings by doing: pg_dump -t 'scores' --schema-only mosaic
# -h aloy-dbsrv

DROP_TABLE = "DROP TABLE IF EXISTS public.similars"

CREATE_TABLE = """CREATE TABLE public.similars (
    inchikey text,
    version text,
    explore_data jsonb
);"""

CREATE_INDEX = """
CREATE INDEX inchikey_similars_idx ON public.similars USING btree (inchikey);
"""

@logged
class Similars(BaseTask):

    def __init__(self, name=None, **params):
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name
        BaseTask.__init__(self, name, **params)

        self.DB = params.get('DB', None)
        if self.DB is None:
            raise Exception('DB parameter is not set')
        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')
        self.cc = ChemicalChecker( self.CC_ROOT )

        c = Config()
        self.hpc_env = c.HPC.system

    def _check_keys_presence_on_spaces( self ):
        metric_obs = None
        metric_prd = None
        map_coords_obs = collections.defaultdict(list)
        dataset_pairs = {}
        for ds in Dataset.get(exemplary=True):
            dataset_pairs[ds.coordinate] = ds.dataset_code
            if metric_obs is None:
                neig1 = self.cc.get_signature("neig1", "reference", ds.dataset_code)
                metric_obs = neig1.get_h5_dataset('metric')[0]
            if metric_prd is None:
                neig3 = self.cc.get_signature("neig3", "reference", ds.dataset_code)
                metric_prd = neig3.get_h5_dataset('metric')[0]
            sign1 = self.cc.get_signature("sign1", "full", ds.dataset_code)
            keys = sign1.unique_keys
            for ik in keys:
                map_coords_obs[ik] += [ds.coordinate]
        return metric_obs, metric_prd, map_coords_obs, dataset_pairs
        
    def _compute_dist_cutoffs( self, dataset_pairs, metric_obs, metric_prd ):
        dss = { 'obs': 'sign1', 'prd': 'sign3' }
        bg_vals = dict()
        signatures = dict()
        for k in dss:
            bg_vals[k] = {}
            signatures[k] = {}
        for coord in dataset_pairs.keys():
            for k in dss:
                cctype = dss[k]
                sign = self.cc.get_signature( cctype, "reference", dataset_pairs[coord])
                bg_vals[k][coord] = sign.background_distances( eval( f"metric_{k}" ) )["distance"]
                signatures[k][coord] = sign
        return bg_vals, signatures            

    def _restore_similar_data_from_chunks( self, host_name,database_name,user_name,database_password, outfile):
        if( os.path.isfile(outfile) ):
            command = 'PGPASSWORD={4} psql -h {0} -d {1} -U {2} -f {3}'\
                      .format(host_name, database_name, user_name, outfile, database_password)
            os.system( command )

    def _import_similar_sql_files(self, keys, sql_path):
        c = Config()
        host = c.DB.host
        user = c.DB.user
        passwd = c.DB.password
        table_new = 'similars'
        db_new = self.DB
        
        keys = set(keys)
        self.__log.info("Importing explore version data")
        for f in tqdm( os.listdir(sql_path) ):
            outfile = os.path.join( sql_path, f )
            self._restore_similar_data_from_chunks( host, db_new, user, passwd, outfile)
        shutil.rmtree( sql_path, ignore_errors=True)
    
    def _custom_chunker(self, keys, additional_data, n_jobs):
        keys = np.array(keys)
        
        dat = {}
        st = 1
        ind = range( len(keys) )
        for ck in np.array_split( ind, n_jobs ):
            idx = str(st)
            dat[idx] = {}
            dat[idx]['keys'] = keys[ck]
            for k in additional_data:
                dat[idx][k] = additional_data[k]
            st += 1
        return dat
    
    def run(self):
        """Run the molecular info step."""
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/similars.py")
        universe_file = os.path.join(self.cachedir, "universe.h5")

        with h5py.File(universe_file, 'r') as hf:
            universe_keys = hf["keys"][:]
        temp = [ k.decode('utf-8') for k in universe_keys ]
        
        try:
            self.__log.info("Creating table")
            #psql.query(DROP_TABLE, self.DB)
            psql.query(CREATE_TABLE, self.DB)
        except Exception as e:
            self.__log.debug("Table similars already exists")
        
        version = self.DB.replace("cc_web_", '')

        # query to see if there is some data filled in new db
        SELECT_CHECK = f"SELECT DISTINCT (inchikey) FROM similars where version = '{ version }';" 
        rows = psql.qstring( SELECT_CHECK, self.DB)
        done = set( [el[0] for el in rows] )
        """
        inprogress = set()
        d = pickle.load( open('/aloy/scratch/ymartins/pipelines/cc_update_2024_02/tmp/jobs_similars_j8j4j9c4/d86515b4-8a12-4a9d-9a0a-f6ac4a693f5d', 'rb') )
        for k in d:
            inprogress.update( d[k]['keys'] )
        d = pickle.load( open('/aloy/scratch/ymartins/pipelines/cc_update_2024_02/tmp/jobs_similars_w4q2pq90/e8fb65ff-5b98-4380-acb4-81cceaac62a6', 'rb') )
        for k in d:
            inprogress.update( d[k]['keys'] )

        done = done.union( inprogress )
        """
        universe_keys = list( set(temp) - done )
        #universe_keys = np.array( universe_keys )

        job_path = None

        if( len(universe_keys) > 0 ):
            # get all bioactive compounds from libraries (with pubchem names)
            lib_bio_file = os.path.join(self.tmpdir, "lib_bio.json")

            # inchikey -> pubchem synonym map, written by the `pubchem` task
            ik_names_file = os.path.join(self.tmpdir, "inchies_names.json")
            if not os.path.isfile(ik_names_file):
                raise Exception(
                    "%s not found! It is produced by the `pubchem` task, "
                    "which must run before `similars`." % ik_names_file)

            self.__log.info("Launching jobs to create json files for " +
                            str(len(universe_keys)) + " molecules")

            job_path = tempfile.mkdtemp(
                prefix='jobs_similars_', dir=self.tmpdir)
            
            sql_path = os.path.join( self.tmpdir, 'temporary_sql' )
            if( not os.path.isdir( sql_path ) ):
                os.mkdir( sql_path )

            nchunks = 1000
            """
            to-do - allow spreading jobs in more than one environment
            ssplit = round( len(universe_keys) * 0.2 )
            if( self.hpc_env == 'sge' ):
                nchunks = 1000
                universe_keys = universe_keys[:ssplit]
            if( self.hpc_env == 'slurm' ):
                nchunks = 1500
                universe_keys = universe_keys[ssplit:]
            """ 
            n_jobs = math.ceil( len(universe_keys) / nchunks ) #500
            if(  len(universe_keys) < nchunks ):
                n_jobs = 1
            metric_obs, metric_prd, map_coords_obs, dataset_pairs = self._check_keys_presence_on_spaces( )
            bg_vals, signatures = self._compute_dist_cutoffs( dataset_pairs, metric_obs, metric_prd )
            vals = ['metric_obs','metric_prd','map_coords_obs','dataset_pairs','bg_vals','signatures']
            additional_data = { 'hpc_env': self.hpc_env  }
            for v in vals:
                additional_data[v] = eval(v)
            custom_elements = self._custom_chunker( universe_keys, additional_data, n_jobs)
            
            params = {}
            params["num_jobs"] = n_jobs # previous was 200
            params["jobdir"] = job_path
            params["job_name"] = "CC_JSONSIM"
            #params["elements"] = universe_keys
            params["custom_chunks"] = custom_elements
            #params["memory"] = 20
            params["cpu"] = 2
            params["mem_by_core"] = 8
            params["time"] = '30-00:00:00' 
            params["wait"] = True
            # job command
            cc_config_path = self.config.config_path
            cc_package = os.path.join(self.config.PATH.CC_REPO, 'package')
            singularity_image = self.config.PATH.SINGULARITY_IMAGE
            command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {} {} {}"
            command = command.format(
                cc_package, cc_config_path, singularity_image, script_path, 
                ik_names_file, lib_bio_file, sql_path, self.DB, version, self.CC_ROOT)
            # submit jobs
            cluster = HPC.from_config(self.config)
            jobs = cluster.submitMultiJob(command, **params)
        
        
        self.__log.info("Checking results")
        #qty_sql_files = len( os.listdir(sql_path) )

        # query to see if there is some data filled in new db
        SELECT_CHECK = f"SELECT DISTINCT (inchikey) FROM similars where version = '{ version }';" 
        rows = psql.qstring( SELECT_CHECK, self.DB)
        done = set( [el[0] for el in rows] )
        qty_keys_inter = done.intersection( set(temp) )
        if( len(qty_keys_inter) == len(temp) ):
        #if( qty_sql_files == n_jobs ):
            # Importing sqls created in each task job
            #self._import_similar_sql_files( universe_keys, sql_path  )

            self.__log.info("Indexing table")
            try:
                psql.query(CREATE_INDEX, self.DB)
            except:
                self.__log.info("Indexes already created")

            if( job_path is not None and os.path.isdir(job_path) ):
                shutil.rmtree(job_path, ignore_errors=True)
            self.mark_ready()
        else:
            e = "Error while saving similars data"
            "There are still keys to calculate the similars"
            self.__log.error(e)
            #raise Exception(e)
            
    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()


