import os
import h5py
import shutil
import tempfile
from shutil import copyfile

from chemicalchecker.core import ChemicalChecker
from chemicalchecker.util.pipeline import BaseTask
from chemicalchecker.util import logged

from chemicalchecker.util.plot import CCStatsPlot


@logged
class StatsPlots(BaseTask):

    def __init__(self, name=None, **params):
        task_id = params.get('task_id', None)
        if task_id is None:
            params['task_id'] = name
        BaseTask.__init__(self, name, **params)

        self.CC_ROOT = params.get('CC_ROOT', None)
        if self.CC_ROOT is None:
            raise Exception('CC_ROOT parameter is not set')

    def run(self):
        """Run the statsplots step."""
        cc = ChemicalChecker(self.CC_ROOT)
        plots_web = os.path.join(self.CC_ROOT, "plots_web")
        if not os.path.exists(plots_web):
            os.mkdir(plots_web)
        plots_dir = os.path.join(plots_web, "plots_stats")
        if not os.path.exists(plots_dir):
            os.mkdir(plots_dir)
        self.__log.info("Stats plots are saved in {}".format(plots_dir))
        try:
            statsplot = CCStatsPlot(cc, width=30, height=30, dpi=200, save=True, save_format='png', save_dir=plots_dir)
            statsplot.plot_all()
            self.mark_ready()
        except Exception as e:
            self.__log.error(
                "Error while creating stats plots")
            if not self.custom_ready():
                raise Exception(e)
            else:
                self.__log.error(e)

    def execute(self, context):
        """Run the statsplots step."""
        self.tmpdir = context['params']['tmpdir']
        self.run()
