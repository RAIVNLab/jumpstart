from torch.optim.lr_scheduler import ChainedScheduler, ConstantLR
import d3rlpy
from dataclasses import dataclass


@dataclass
class ChainedSchedulerFactory(d3rlpy.optimizers.LRSchedulerFactory):
    def __init__(self, factory1=None, factory2=None):
        self.factory1 = factory1
        self.factory2 = factory2

    def create(self, optim):
        if self.factory1 is None or self.factory2 is None:
            return ConstantLR(optim, factor=1.0)
        # Create the two PyTorch schedulers
        sched1 = self.factory1.create(optim)
        sched2 = self.factory2.create(optim)
        # Chain them together
        return ChainedScheduler([sched1, sched2], optimizer=optim)

    @staticmethod
    def get_type() -> str:
        return "chained"


# register
d3rlpy.optimizers.lr_schedulers.register_lr_scheduler_factory(ChainedSchedulerFactory)
