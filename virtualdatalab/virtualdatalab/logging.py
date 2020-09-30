import logging
import sys

stdout_level = logging.WARNING

def getLogger(module_name, stdout=None):
    format = '%(asctime)s [%(name)s:%(levelname)s] %(message)s'
    logging.basicConfig(filename='virtualdatalab.log',
                        level=logging.DEBUG,
                        filemode='w',
                        format=format,
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(module_name)

    global stdout_level
    if stdout:
        stdout_level = stdout
    # Add handler for stdout
    if stdout_level != logging.WARNING:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(stdout_level)
        formatter = logging.Formatter(format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
