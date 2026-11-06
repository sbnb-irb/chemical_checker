"""Pipeline to generate a full CC update.
steps (a.k.a. tasks) for a full CC update are the following:

1. Download Datasources (raw input including any external DB)
2. Generate Molrepos (parse downloaded files and save molecules identifiers)
3. Generate Validation sets (MoA and ATC)
4. Compute Sign0
4.1. Sign0 for B,C,D,E levels (CC universe definition)
4.2. Calculate A level (chemistry properties)
4.3. Sign0 for A level
5. Compute Sign1 (and neig clus and proj)
6. Compute Sign2 (and neig clus and proj)
7. Generate stacked Sign2 for universe
8. Compute Sign3
9. Create symlinks

**N.B.**
Once the jobs for a task are completed we can evaluate task
performances (i.e. `cpu` and `memory` parameters were ok?) with:

`ssh pac qacct -j JOBID|egrep 'taskid|maxvmem|ru_wallclock|cpu'`

* 'taskid' the qacct give result aggregated, this help breaking down results.
* 'maxvmem' tell us about max RAM used (tweak `memory` parameter).
* 'ru_wallclock' gives the total time the job took.
* 'cpu' gives the cpu times consumed.

the ratio 'cpu/ru_wallclock' should be as close as possible to the
requested cpus (tweak `cpu` parameter).
"""
import os
import sys
import json
import logging
import argparse
import tempfile
from update_resources import generate_chembl_files, fetch_pharmacodb

from chemicalchecker import ChemicalChecker
from chemicalchecker.core import Validation
from chemicalchecker.core.sign3 import sign3
from chemicalchecker.util import Config, HPC, logged
from chemicalchecker.database import Dataset, Datasource
from chemicalchecker.database import Molrepo, Calcdata
from chemicalchecker.util.pipeline import Pipeline, PythonCallable
from chemicalchecker.util.pipeline import CCFit, CCPredict
from update_resources.create_database import create_db_dataset
from chemicalchecker.core.diagnostics import Diagnosis

# this is needed to export signaturizers at the end
from signaturizer.exporter import export_batch


def pipeline_parser():
    """Parse pipeline arguments."""
    description = 'Run the full CC update pipeline.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        'cc_root', type=str,
        help='Directory where the new CC instance will be generated '
        '(e.g. `/aloy/web_checker/package_cc/2020_01`)')
    parser.add_argument(
        'pipeline_dir', type=str,
        help='Directory where the pipeline will run '
        '(e.g. `/aloy/scratch/mbertoni/pipelines/cc_update_2020_01`)')
    parser.add_argument(
        '-p', '--signaturizer_version_path', type=str, default="/aloy/web_checker/signaturizers/", required=False,
        help='Path to the new version to export signaturizers '
        '(i.e. /aloy/web_checker/signaturizers/ ).')
    parser.add_argument(
        '-r', '--reference_cc', type=str, default="", required=False,
        help='Root dir of the CC instance to use as reference '
        '(i.e. triplet sampling in sign0).')
    parser.add_argument(
        '-m', '--mode_complete_universe', type=str, default="off",
        required=False, choices=['off', 'full', 'fast'],
        help='Fill the A-level (chemistry) gaps of the sign2 universe: '
        '`off` skips the task entirely (default), `full` completes all A '
        'spaces, `fast` skips A2.001. NB: sign3 is configured with '
        '`complete_universe: False` and reads the *uncompleted* universe, so '
        '`full`/`fast` currently produce a ~40GB `.complete.h5` pair that '
        'nothing downstream opens.')
    parser.add_argument(
        '-t', '--only_tasks', type=str, nargs="+", default=[],
        required=False,
        help='Names of tasks that will `exclusively` run by the pipeline.')  # format like -t sign0 sign1 sign2
    parser.add_argument(
        '-s', '--exclude_tasks', type=str, nargs="+", default=[],
        required=False,
        help='Names of tasks that will be skipped.')  # format like -t sign0 sign1 sign2
    parser.add_argument(
        '-c', '--config', type=str, required=False,
        default=os.environ["CC_CONFIG"],
        help='Config file to be used. If not specified CC_CONFIG enviroment'
        ' variable is used.')
    parser.add_argument(
        '-d', '--dry_run', action='store_true',
        help='Execute pipeline script without running the pipeline.')
    return parser


@logged(logging.getLogger("[ PIPELINE %s ]" % os.path.basename(__file__)))
def main(args):
    # initialize Pipeline
    cfg = Config(args.config)
    pp = Pipeline(pipeline_path=args.pipeline_dir, keep_jobs=True,
                  config=cfg, only_tasks=args.only_tasks,
                  exclude_tasks=args.exclude_tasks)

    # print arguments
    for arg in vars(args):
        main._log.info('[ ARGS ] {:<25s}: {}'.format(arg, getattr(args, arg)))

    fit_order = ['sign0', 'sign1', 'clus1', 'proj1', 'neig1',
                 'sign2', 'clus2', 'proj2', 'neig2', 'sign3',
                 'clus3', 'proj3', 'neig3', 'sign4', 'clus4',
                 'proj4', 'neig4']

    data_calculators = [
        'morgan_fp_r2_2048',
        'e3fp_3conf_1024',
        'murcko_1024_cframe_1024',
        'maccs_keys_166',
        'general_physchem_properties',
        #'chembl_target_predictions_v23_10um'
    ]

    validation_sets = ['moa', 'atc']

    # HPC parameters and resources observations
    # NB: slurm.py/sge.py only honor 'mem_by_core' (emitted as
    # --mem-per-cpu); the plain 'memory' kwarg is dead code. Total job
    # memory = cpu * mem_by_core. Values below preserve the previous totals
    # (with headroom over the observed peaks noted in the comments).
    hpc_kwargs = {
        # sign0
        #  is using the most memory with ~GB
        #  is the one taking longer with ~s (h)
        'sign0': {'mem_by_core': 30, 'cpu': 4},
        # sign1 (w/o metric_learning) does not parallelize
        # B4 historically peaked ~12GB, but the larger 2026 sources OOM-killed
        # the LSI predict step at the old 16GB ceiling, so bump to 32GB.
        'sign1': {'mem_by_core': 8, 'cpu': 4},     # 32GB
        # sign2 paralelize well and requires memory
        # A2 is using the most memory with ~42GB
        # A1 is the one taking longer with ~158000s (43h)
        'sign2': {'mem_by_core': 4, 'cpu': 16},    # 64GB
        # sign3
        # A1 is using the most memory with ~59GB  <- 1.0M universe (2021_07)
        # A1 is the one taking longer with ~186000s (52h)
        # 2026_04 (1.54M universe). Bumped to 224GB.
        'sign3': {'mem_by_core': 28, 'cpu': 8},    # 224GB
        # sign4
        'sign4': {'mem_by_core': 10, 'cpu': 20},   # 200GB
        #'sign4': {'memory': 100, 'cpu': 8},

        # neig1 paralelize very well and require very few memory
        # A2 is the one taking longer with ~9100s (2.5h)
        'neig1': {'mem_by_core': 2, 'cpu': 16},    # 32GB — bumped for 2026 scale
        'neig2': {'mem_by_core': 1, 'cpu': 16},
        'neig3': {'mem_by_core': 1, 'cpu': 16},
        'neig4': {'mem_by_core': 1, 'cpu': 16},
        # clus1 does not paralelize very well and require memory
        # A1 is using the most memory ~28Gb
        # A1 is the one taking longer with ~10700s (3h)
        'clus1': {'mem_by_core': 12, 'cpu': 4},    # 48GB — bumped; 2026 sources OOM at 32GB
        'clus2': {'mem_by_core': 8, 'cpu': 4},
        'clus3': {'mem_by_core': 8, 'cpu': 4},
        'clus4': {'mem_by_core': 8, 'cpu': 4},
        # proj1 paralelize very well and require few memory
        # A1 is using the most memory ~13Gb
        # A1 is the one taking longer with ~4500s (1.5h)
        'proj1': {'mem_by_core': 4, 'cpu': 16},    # 64GBs
        'proj2': {'mem_by_core': 2, 'cpu': 16},
        'proj3': {'mem_by_core': 2, 'cpu': 16},
        'proj4': {'mem_by_core': 2, 'cpu': 16}
    }

    # on which signature molset to call the fit?
    molset = {
        'sign0': 'full',
        'sign1': 'full',
        'sign2': 'reference',
        'sign3': 'full',
        'sign4': 'full',
        'neig1': 'reference',
        'neig2': 'reference',
        'neig3': 'reference',
        'neig4': 'reference',
        'clus1': 'reference',
        'clus2': 'reference',
        'clus3': 'reference',
        'clus4': 'reference',
        'proj1': 'reference',
        'proj2': 'reference',
        'proj3': 'reference',
        'proj4': 'reference'
    }

    # TASK: Create new CC DB
#    def create_db():
        # Create a new database for the current CC update
    create_db_dataset()

#    creating_db_task = PythonCallable(name="creating_db", python_callable=create_db)
#    pp.add_task(creating_db_task)
    # END TASK --- i ut in here because the next line needs the table dataset and the db is empty

    # initialize parameter holders (two dict, one for init and one for fit)
    datasets = [ds.code for ds in Dataset.get(exemplary=True)]
    sign_kwargs = {}
    fit_kwargs = {}
    for cctype in fit_order:
        fit_kwargs[cctype] = {}
        sign_kwargs[cctype] = {}

    # GENERAL FIT/INIT PARAMETERS
    cc = ChemicalChecker(args.cc_root)
    sign2_universe = os.path.join(pp.cachedir, "sign2_universe_stacked.h5")
    sign2_coverage = os.path.join(pp.cachedir, "sign2_universe_coverage.h5")
    sign2_list = [cc.get_signature('sign2', 'full', ds)
                  for ds in cc.datasets_exemplary()]
    mfp = cc.get_signature('sign0', 'full', 'A1.001').data_path
   
    for ds in datasets:
        fit_kwargs['sign0'][ds] = {
            'key_type': 'inchikey',
            'do_triplets': False,
            'sanitize': True,
            'validations': True,
            'diagnostics': False,
            'sanitizer_kwargs': {'max_features': 10000,'chunk_size': 10000}
        }
        fit_kwargs['sign1'][ds] = {
            'metric_learning': False,
            'diagnostics': False
        }
        fit_kwargs['sign2'][ds] = {
            'validations': True,
            'diagnostics': False,
            # Keep node2vec single-threaded: its Hogwild SGD degrades the
            # embedding (norm inflation, loss of MoA/ATC structure) when run
            # with many threads on the multi-core sign2 nodes. AdaNet still
            # uses the full CPU allocation below.
            'node2vec_kwargs': {'cpu': 1},
            'adanet_kwargs': {'cpu': hpc_kwargs['sign2']['cpu']}
        }
        fit_kwargs['sign3'][ds] = {
            'sign2_list': sign2_list,
            'complete_universe': False,
            'sign2_universe': sign2_universe,
            'sign2_coverage': sign2_coverage,
            'sign0': mfp,
            'diagnostics': False
        }
        sign_kwargs['sign3'][ds] = {
            'sign2': {'cpu': hpc_kwargs['sign3']['cpu']},
            'hpc_args': hpc_kwargs['sign3']
        }
        sign_kwargs['neig1'][ds] = {
            'cpu': hpc_kwargs['neig1']['cpu']
        }
        sign_kwargs['neig2'][ds] = {
            'cpu': hpc_kwargs['neig2']['cpu']
        }
        sign_kwargs['clus1'][ds] = {
            'cpu': hpc_kwargs['clus1']['cpu'],
            'general_params': {'balance': 1.5}
        }
        sign_kwargs['clus2'][ds] = {
            'cpu': hpc_kwargs['clus2']['cpu']
        }
        sign_kwargs['proj1'][ds] = {
            'cpu': hpc_kwargs['proj1']['cpu']
        }
        sign_kwargs['proj2'][ds] = {
            'cpu': hpc_kwargs['proj2']['cpu']
        }

    # DATASET SPECIFIC FIT/INIT PARAMETERS
    # we want to keep exactly 2048 features (Morgan fingerprint) for A1
    fit_kwargs['sign0']['A1.001'] = {'sanitize': False}


    # TASK: Download all datasources
    def download_fn(tmpdir):
        # Generate the Chembl files drugtargets and drugindications via
        # Chembl Python API in /aloy/web_checker/repo_data
        generate_chembl_files()

        # Generate the PharmacoDB postgreSQL database via API queries
        fetch_pharmacodb()

        job_path = tempfile.mkdtemp(prefix='jobs_download_', dir=tmpdir)
        # start download jobs (one per Datasource)
        # job will wait until finished
        job = Datasource.download_hpc(job_path, only_essential=True)
        if job.status() == HPC.ERROR:
            main._log.error(
                "There are errors in some of the downloads jobs")
        
        flagDownload = Datasource.test_all_downloaded(only_essential=True)
        # check if the downloads are really done
        if not flagDownload:
            main._log.error(
                "Something went WRONG while DOWNLOADING, please retry")
            # print the faulty one
            missing_datasources = set()
            for ds in Datasource.get():
                for dset in ds.datasets:
                    if dset.essential:
                        missing_datasources.add(ds)
                        break
                for molrepo in ds.molrepos:
                    if molrepo.essential:
                        missing_datasources.add(ds)
                        break
            for ds in missing_datasources:
                if not ds.available:
                    main._log.error("Datasource %s not available" % ds)
            raise Exception('Not all datasources were downloaded correctly')
    
    flagDownload = Datasource.test_all_downloaded(only_essential=True)
    if( not flagDownload ):
        downloads_task = PythonCallable(name="downloads",
                                    python_callable=download_fn,
                                    op_args=[pp.tmpdir])
        pp.add_task(downloads_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Generate Molrepos
    molrepos_task = PythonCallable(name="molrepos",
                                   #python_callable=Molrepo.molrepo_hpc,
                                   python_callable=Molrepo.molrepo_sequential,
                                   op_args=[pp.tmpdir],
                                   op_kwargs={'only_essential': True})
    pp.add_task(molrepos_task)
    
    # END TASK
    #############################################

    #############################################
    # TASK: Generate validation sets
    def create_val_set_fn(set_name):
        main._log.info("Creating validation set for " + set_name)
        print("Creating validation set for " + set_name)
        val = Validation(Config(args.config).PATH.validation_path, set_name)
        try:
            val.run()
        except Exception as ex:
            print(ex)
            raise Exception("Validation set '%s' not working" % set_name)

    for val_set in validation_sets:
        val_set_task = PythonCallable(
            name="val_set_" + val_set,
            python_callable=create_val_set_fn,
            op_args=[val_set])
        pp.add_task(val_set_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 0 for derived spaces only (i.e. BCDE levels)
    cctype = 'sign0'
    dss = [ds.code for ds in Dataset.get(exemplary=True) if ( ds.derived and not ds.code.startswith('D1') ) ]
    task = CCFit(args.cc_root, cctype, molset[cctype],
                 datasets=dss,
                 name = 'sign0_BCDE',
                 fit_kwargs=fit_kwargs[cctype],
                 sign_kwargs=sign_kwargs[cctype],
                 hpc_kwargs=hpc_kwargs[cctype])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate diagnostics plots of sign0 for derived spaces: the reference cc version is provided
    # by the 'reference_cc' input parameter (only for sign0 case, for other signatures 
    # the reference is args.cc_root itself)
    cctype = 'sign0'
    task = PythonCallable(name="diagnostics_sign0_BCDE",
                         python_callable=Diagnosis.diagnostics_hpc,
                         op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.reference_cc],
                         op_kwargs={'cc_config': args.config})
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate data (defined for universe)
    def calculate_data_fn(type_data, tmpdir, inchikey_inchi):
        main._log.info("Calculating data for " + type_data)
        job_path = tempfile.mkdtemp(
            prefix='jobs_molprop_' + type_data + "_", dir=tmpdir)
        calculator = Calcdata(type_data)
        # This method sends the job and waits for the job to finish
        calculator.calcdata_hpc(job_path, list(inchikey_inchi), cc_config=args.config)
        missing = len(calculator.get_missing_from_set(inchikey_inchi))
        if missing > 0:
            raise Exception(
                "Not all molecular properties were calculated. There are " +
                str(missing) + " missing out of " + str(len(inchikey_inchi)))

    # after running the first tranche of sign0 we known the CC universe
    universe = cc.universe
    main._log.info('CC Universe will include %s molecules.' % len(universe))
    
    # Restarting from where it stopped
    for data_calc in data_calculators:
        calc_data_task = PythonCallable(
            name="calc_data_" + data_calc,
            python_callable=calculate_data_fn,
            op_args=[data_calc, pp.tmpdir, universe])
        pp.add_task(calc_data_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 0 for non derived spaces only (i.e. A level)
    cctype = 'sign0'
    dss = [ds.code for ds in Dataset.get(exemplary=True) if not ds.derived]
    task = CCFit(args.cc_root, cctype, molset[cctype],
                 datasets=dss,
                 name = 'sign0_A',
                 fit_kwargs=fit_kwargs[cctype],
                 sign_kwargs=sign_kwargs[cctype],
                 hpc_kwargs=hpc_kwargs[cctype])
    pp.add_task(task)
    # END TASK
    #############################################
    
    
    #############################################
    # TASK: Calculate diagnostics plots of sign0 for A* spaces: the reference cc version is provided
    # by the 'reference_cc' input parameter (only for sign0 case, for other signatures 
    # the reference is args.cc_root itself)
    cctype = 'sign0'
    task = PythonCallable(name="diagnostics_sign0_A",
                         python_callable=Diagnosis.diagnostics_hpc,
                         op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.reference_cc],
                         op_kwargs={'cc_config': args.config})
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 1-2 also clus, proj and neig
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    s0_idx = fit_order.index('sign0') + 1
    s3_idx = fit_order.index('sign3')
    for cctype in fit_order[s0_idx:s3_idx]:
        task = CCFit(args.cc_root, cctype, molset[cctype],
                     datasets=dss,
                     fit_kwargs=fit_kwargs[cctype],
                     sign_kwargs=sign_kwargs[cctype],
                     hpc_kwargs=hpc_kwargs[cctype])
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Pre-build the sign metadata cache (single writer)
    # The diagnostics tasks below run as parallel Slurm arrays and each reads
    # cc_root's `metadata.json.zip` via `sign_metadata()`. That cache is built
    # lazily on first access; letting the array build it concurrently races
    # (readers hit a half-written file -> JSONDecodeError). Build it once here
    # on the head node so every diagnostics job only ever reads a complete file.
    def prebuild_metadata_fn():
        cc._get_metadata(overwrite=True)

    prebuild_metadata_task = PythonCallable(
        name="prebuild_metadata",
        python_callable=prebuild_metadata_fn)
    pp.add_task(prebuild_metadata_task)
    # END TASK
    #############################################

    #############################################
    # TASK: diagonistc plots for sign1-sign2
    cctypes = ['sign1', 'sign2']
    for cctype in cctypes:
        task = PythonCallable(name="diagnostics_" + cctype,
                            python_callable=Diagnosis.diagnostics_hpc,
                            op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.cc_root],
                            op_kwargs={'cc_config': args.config})
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Cache sign2 universe
    def sign2_universe_fn(sign2_list, sign2_universe, sign2_coverage):
        # FIXME this should be performed in a HPC task
        # generate sign2 universes (sign3 specific pre-calculations)
        if not os.path.isfile(sign2_universe):
            sign3.save_sign2_universe(sign2_list, sign2_universe)
        if not os.path.isfile(sign2_coverage):
            sign3.save_sign2_coverage(sign2_list, sign2_coverage)

    sign2_universe_task = PythonCallable(
        name="sign2_universe",
        python_callable=sign2_universe_fn,
        op_args=[sign2_list, sign2_universe, sign2_coverage])
    pp.add_task(sign2_universe_task)
    # END TASK
    #############################################
    
    #############################################
    # TASK: Complete universe, calculating physical-chemical features for the molecules that exist in the other cc spaces but are not in the original set of the A spaces.
    tmpCC = pp.tmpdir
    sign2_src_dataset_list = [sign.dataset for sign in sign2_list]
    
    refcc = None
    if( args.reference_cc != "" ):
        refcc = args.reference_cc
    
    calc_idx_chemical_spaces=[0,1,2,3,4]
    if( args.mode_complete_universe == 'fast' ):
        calc_idx_chemical_spaces=[0,2,3,4]    
    
    task = PythonCallable(name="complete_sign2_universe_global",
                         python_callable = sign3.complete_sign2_universe_global_pipeline,
                         op_args=[sign2_universe, sign2_coverage],
                         op_kwargs={'tmp_path': pp.tmpdir, 'root_cc': args.cc_root, 'ref_cc': refcc, 'calc_idx_chemical_spaces': calc_idx_chemical_spaces, 'sign2_src_dataset_list': sign2_src_dataset_list })
    # This completion has never run in any shipped CC version (paper, 2020_02,
    # 2021_07, 2024_02): its output is written to `<universe>.complete.h5`, but
    # sign3 above is configured with `complete_universe: False` and is handed
    # the *original* universe paths, so nothing ever opens the completed files.
    # Skipped by default; pass `-m full`/`-m fast` to run it anyway.
    if args.mode_complete_universe == 'off':
        main._log.info("Task SKIPPED: 'complete_sign2_universe_global' "
                       "(mode_complete_universe='off')")
    else:
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 3
    cctype = 'sign3'
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    task = CCFit(args.cc_root, cctype, molset[cctype],
                 datasets=dss,
                 fit_kwargs=fit_kwargs[cctype],
                 sign_kwargs=sign_kwargs[cctype],
                 hpc_kwargs=hpc_kwargs[cctype])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate diagnostics plots of sign3 for all spaces
    cctype = 'sign3'
    task = PythonCallable(name="diagnostics_" + cctype,
                         python_callable=Diagnosis.diagnostics_hpc,
                         op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.reference_cc],
                         op_kwargs={'cc_config': args.config})
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate clus3, proj3 and neig3
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    s3_idx = fit_order.index('sign3') + 1
    s4_idx = fit_order.index('sign4')
    for cctype in fit_order[s3_idx:s4_idx]:
        task = CCFit(args.cc_root, cctype, molset[cctype],
                     datasets=dss,
                     fit_kwargs=fit_kwargs[cctype],
                     sign_kwargs=sign_kwargs[cctype],
                     hpc_kwargs=hpc_kwargs[cctype])
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate signatures 4
    cctype = 'sign4'
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    task = CCFit(args.cc_root, cctype, molset[cctype],
                 datasets=dss,
                 fit_kwargs=fit_kwargs[cctype],
                 sign_kwargs=sign_kwargs[cctype],
                 hpc_kwargs=hpc_kwargs[cctype])
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate diagnostics plots of sign4 for all spaces
    cctype = 'sign4'
    task = PythonCallable(name="diagnostics_" + cctype,
                         python_callable=Diagnosis.diagnostics_hpc,
                         op_args=[pp.tmpdir, args.cc_root, cctype, molset[cctype], dss, args.reference_cc],
                         op_kwargs={'cc_config': args.config})
    pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Calculate clus4, proj4 and neig4
    dss = [ds.code for ds in Dataset.get(exemplary=True)]
    s4_idx = fit_order.index('sign4') + 1
    for cctype in fit_order[s4_idx:]:
        task = CCFit(args.cc_root, cctype, molset[cctype],
                     datasets=dss,
                     fit_kwargs=fit_kwargs[cctype],
                     sign_kwargs=sign_kwargs[cctype],
                     hpc_kwargs=hpc_kwargs[cctype])
        pp.add_task(task)
    # END TASK
    #############################################

    #############################################
    # TASK: Create sym links for exemplary signatures
    def create_symlinks_fn(cc, sign_ref):
        # link to exemplary signatures
        target_path = os.path.join(args.cc_root, "exemplary")
        if not os.path.isdir(target_path):
            os.mkdir(target_path)
        for ds in Dataset.get(exemplary=True):
            signature_path = cc.get_signature_path(sign_ref, "full", ds.code)
            source_path = signature_path[:-6]
            target_dir = os.path.join(target_path, ds.code[:1])
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)
            if not os.path.exists(os.path.join(target_dir, ds.code[:2])):
                os.symlink(source_path, os.path.join(target_dir, ds.code[:2]))
        # link to single folder for all signatures
        cc.export_symlinks()
        # add metadata to signatures
        cc.add_sign_metadata(signature='sign*')

    symlinks_task = PythonCallable(
        name="symlinks",
        python_callable=create_symlinks_fn,
        op_args=[cc, 'sign1'])
    pp.add_task(symlinks_task)
    # END TASK
    #############################################

    #############################################
    # TASK: Export signaturizers
    def export_signaturizers(cc_root, path='/aloy/web_checker/signaturizers/'):
        cc = ChemicalChecker(args.cc_root)
        sign_path = os.path.join(path, cc.name)
        if not os.path.isdir(sign_path):
            os.mkdir(sign_path)
        export_batch(cc, sign_path)

    export_task = PythonCallable(name="export_signaturizers",
                                 python_callable=export_signaturizers,
                                 op_args=[args.cc_root],
                                 op_kwargs= { 'path': args.signaturizer_version_path } )
    pp.add_task(export_task)
    # END TASK
    #############################################
    
    #############################################
    # RUN the pipeline!
    main._log.info('TASK SEQUENCE: %s' % ', '.join([t.name for t in pp.tasks]))
    if not args.dry_run:
        pp.run()
    # END PIPELINE
    #############################################


if __name__ == '__main__':
    # parse arguments
    args = pipeline_parser().parse_args(sys.argv[1:])
    main(args)
