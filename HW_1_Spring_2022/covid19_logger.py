from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class Covid19Logger:
    def __init__(self, opt):
        self.opt = opt
        self.logger = self._load_logger()

    def _load_logger(self):
        now = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        log_dir = os.path.join(self.opt.log_dir, now + "_" + self.opt.model)
        logger = SummaryWriter(log_dir=log_dir)
        for k in self.opt.__dict__.keys():
            logger.add_text(k, str(self.opt.__dict__[k]))
        print('Log files saved to ', logger.file_writer.get_logdir())
        return logger

    def get_logger(self):
        return self.logger
