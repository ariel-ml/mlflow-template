import logging
import logging.config
import json


def getLogger(name: str):
    """
    Initialize the logging module with the configuration given in the
    logging_config.json file and return a logger with the given name.

    Parameters
    ----------
    name : str
        The name of the logger to return.

    Returns
    -------
    logger : logging.Logger
        The logger with the given name.
    """
    with open("logging_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        logging.config.dictConfig(config)
    return logging.getLogger(name)
