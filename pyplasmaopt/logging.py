import sys
import logging
import os
from mpi4py import MPI
comm = MPI.COMM_WORLD

__all__ = ("debug", "info", "warning", "error", "set_file_logger")

logger = logging.getLogger('PyPlasmaOpt')

handler = logging.StreamHandler()
formatter = logging.Formatter(fmt="%(levelname)s %(message)s")
handler.setFormatter(formatter)
if comm is not None and comm.rank != 0:
    handler = logging.NullHandler()
logger.addHandler(handler)

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level
    
    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)
    
    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def set_file_logger(path):
    filename, file_extension = os.path.splitext(path)
    from math import log10, ceil
    digits = ceil(log10(comm.size))
    fileHandler = logging.FileHandler(filename + "-rank" + ("%i" % comm.rank).zfill(digits) + file_extension, mode='w')
    formatter = logging.Formatter(fmt="%(asctime)s:%(name)s:%(levelname)s %(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

logger.setLevel(logging.INFO)

debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error

sys.stdout = LoggerWriter(logger.warning)
sys.stderr = LoggerWriter(logger.warning)
