import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.base import rank_zero_experiment
from collections import defaultdict

class History(object):
    def __init__(self):
        self.steps = []
        self.values = []
    def append(self, step, value):
        self.steps.append(step)
        self.values.append(value)

class LogHistory(LightningLoggerBase):
    
    def __init__(self) -> None:
        # initialize logger
        super(LogHistory, self).__init__()
        # create a history dict to store all metrics
        self.history = defaultdict(History)

    @property
    def name(self) -> str:
        return "Custom Logger"

    @property
    def version(self) -> str:
        return "0.1"

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # remove metrics that we don't care for
        metrics.pop("epoch")
        # add each metric to the history
        for name, value in metrics.items():
            self.history[name].append(step, value)

    @rank_zero_only
    def save(self):
        super(LogHistory, self).save()

    def __getitem__(self, name):
        return self.history[name]

