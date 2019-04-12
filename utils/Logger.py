import logging
import os


class Logger(object):
    """
    set logger

        https://www.cnblogs.com/CJOKER/p/8295272.html

    """

    def __init__(self, logger_path):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self.logfile = logging.FileHandler(logger_path)
        #
        self.logfile.setLevel(logging.DEBUG)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(
            '%(asctime)s -%(filename)s:%(lineno)s - %(levelname)s - %(message)s')
        self.logfile.setFormatter(formatter)
        self.logdisplay = logging.StreamHandler()
        #
        self.logdisplay.setLevel(logging.DEBUG)
        self.logdisplay.setFormatter(formatter)
        self.logger.addHandler(self.logfile)
        self.logger.addHandler(self.logdisplay)

    def get_logger(self):
        return self.logger
