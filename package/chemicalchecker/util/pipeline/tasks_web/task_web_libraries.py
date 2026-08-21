import os
import json
import shutil
import tempfile

from chemicalchecker.util import psql
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged, HPC

# We got these strings by doing: pg_dump -t 'scores' --schema-only mosaic
# -h aloy-dbsrv

DROP_TABLE_DESC = "DROP TABLE IF EXISTS public.library_description"

DROP_TABLE = "DROP TABLE IF EXISTS public.libraries"

CREATE_TABLE = """CREATE TABLE public.libraries (
    inchikey text,
    lib text,
    is_bioactive smallint,
    is_landmark smallint
);"""

CREATE_TABLE_DESC = """CREATE TABLE public.library_description (
    lib text NOT NULL PRIMARY KEY,
    name text,
    description text,
    urls text,
    parser text,
    rank integer
);"""

CREATE_INDEX = """
CREATE INDEX inchikey_libraries_idx ON public.libraries USING btree (inchikey);
CREATE INDEX is_bioactive_libraries_idx ON public.libraries USING btree (is_bioactive);
CREATE INDEX is_landmark_libraries_idx ON public.libraries USING btree (is_landmark);
CREATE INDEX lib_libraries_idx ON public.libraries USING btree (lib);
ALTER TABLE ONLY public.libraries ADD CONSTRAINT libraries_lib_fkey FOREIGN KEY (lib) REFERENCES public.library_description(lib);
"""

CREATE_INDEX_DESC = """
CREATE INDEX lib_library_description_idx ON public.library_description USING btree (lib);
CREATE INDEX name_library_description_idx ON public.library_description USING btree (name);
CREATE INDEX rank_library_description_idx ON public.library_description USING btree (rank);
"""

INSERT_DESC = "INSERT INTO library_description VALUES %s"

CHECK = "select distinct(lib) from libraries"


@logged
class Libraries(BaseTask):

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
        self.libraries = params.get('libraries', None)
        if self.libraries is None:
            raise Exception('libraries parameter is not set')
    
    def _prepare_lib_file(self ):
        # get all bioactive compounds from libraries (with pubchem names)
        lib_bio_file = os.path.join(self.tmpdir, "lib_bio.json")
        if not os.path.exists(lib_bio_file):
            text_bio = "select  library_description.name as lib,lib.inchikey from libraries as lib INNER JOIN library_description on lib.lib = library_description.lib   where lib.is_bioactive = '1' order by  library_description.rank"
            lib_bio = psql.qstring(text_bio, self.DB)
            ref_bioactive = dict()
            for lib in lib_bio:
                if lib[0] not in ref_bioactive:
                    ref_bioactive[lib[0]] = set()
                ref_bioactive[lib[0]].add(lib[1])
            for lib in ref_bioactive:
                ref_bioactive[lib] = list(ref_bioactive[lib])
            with open(lib_bio_file, 'w') as outfile:
                json.dump(ref_bioactive, outfile)
    
    def run(self):
        """Run the molecular info step."""
        script_path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), "scripts/libraries.py")

        universe_file = os.path.join(self.cachedir, "universe.h5")
        try:
            self.__log.info("Creating table")
            psql.query(DROP_TABLE, self.DB)
            psql.query(CREATE_TABLE, self.DB)
            psql.query(DROP_TABLE_DESC, self.DB)
            psql.query(CREATE_TABLE_DESC, self.DB)
        except Exception as e:
            self.__log.error("Error while creating libraries table")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)
                return

        i = 1
        for idname, v in self.libraries.items():
            s = "('%s',%s,%d)" % (idname, ",".join(["'%s'" % x for x in v]), i)
            psql.query(INSERT_DESC % s, self.DB)
            i += 1

        self.__log.info("Generating libraries for " +
                        str(len(self.libraries.keys())) + " libraries")
        job_path = tempfile.mkdtemp(
            prefix='jobs_libraries_', dir=self.tmpdir)
        libraries_path = os.path.join(self.tmpdir, "libraries_files")
        if not os.path.exists(libraries_path):
            original_umask = os.umask(0)
            os.makedirs(libraries_path, 0o775)
            os.umask(original_umask)

        params = {}
        params["num_jobs"] = len(self.libraries.keys())
        params["jobdir"] = job_path
        params["job_name"] = "CC_LIBRARIES"
        params["elements"] = self.libraries
        params["wait"] = True
        params["cpu"] = 1
        params["mem_by_core"] = 16
        # job command
        cc_config_path = self.config.config_path
        cc_package = os.path.join(self.config.PATH.CC_REPO, 'package')
        singularity_image = self.config.PATH.SINGULARITY_IMAGE
        command = "SINGULARITYENV_PYTHONPATH={} SINGULARITYENV_CC_CONFIG={} singularity exec {} python {} <TASK_ID> <FILE> {} {} {} {}"
        command = command.format(
            cc_package, cc_config_path, singularity_image, script_path, universe_file, libraries_path, self.DB, self.CC_ROOT)
        # submit jobs
        cluster = HPC.from_config(self.config)
        jobs = cluster.submitMultiJob(command, **params)

        try:
            self.__log.info("Checking table")
            libs = psql.qstring(CHECK, self.DB)
            if len(libs) != len(self.libraries.keys()):
                if not self.custom_ready():
                    raise Exception(
                        "Not all libs were added to libraries (%d/%d)" % (len(libs), len(self.libraries.keys())))
                else:
                    self.__log.error(
                        "Not all libs were added to libraries (%d/%d)" % (len(libs), len(self.libraries.keys())))
            else:
                self.__log.info("Indexing table")
                psql.query(CREATE_INDEX, self.DB)
                psql.query(CREATE_INDEX_DESC, self.DB)
                
                # get all bioactive compounds from libraries (with pubchem names)
                self._prepare_lib_file()
                
                shutil.rmtree(job_path, ignore_errors=True)
                self.mark_ready()
        except Exception as e:
            self.__log.error("Error while checking libraries table")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)

    def execute(self, context):
        """Run the molprops step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
