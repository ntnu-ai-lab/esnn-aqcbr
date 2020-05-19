import logging
import os
import sys
from logging.handlers import WatchedFileHandler


def getlogger(logkey, logfile):
    logger = logging.getLogger(logkey)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s '
                                  + '- %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    logfilehandler = WatchedFileHandler(
        os.environ.get("LOGFILE", logfile))
    logfilehandler.setFormatter(formatter)

    logger.addHandler(logfilehandler)

    return logger


class LoggingUtility(object):
    __instances = {}
    @staticmethod
    def getInstance(key):
        """This method gives you an instance of a logger to a key, which logs to
        stdout using tag "key" and also to a file kalled key.log. If this
        logger already exists, you get that instance, if not, the method
        creates a new instance for that key and returns that.

        :param key: string that acts as key, if you want to log "vg" etc..
        :return: returns a logger instance for that key

        """
        if LoggingUtility.__instances is None:
            LoggingUtility()
        if key not in LoggingUtility.__instances or \
           LoggingUtility.__instances[key] is None:
            LoggingUtility.__instances[key] = getlogger(key, f"{key}.log")
        return LoggingUtility.__instances[key]

    @staticmethod
    def setLogLevel(key, loglevel):
        instance = LoggingUtility.getInstance(key)
        for handler in instance.handlers:
            handler.setLevel(loglevel)
        instance.setLevel(loglevel)

    def __init__(self):
        if LoggingUtility.__instances is not None:
            raise Exception("This class is a singleton!")
        else:
            LoggingUtility.__instances = dict()
