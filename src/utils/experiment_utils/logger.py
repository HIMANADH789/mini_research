import logging

def get_logger(log_path):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger