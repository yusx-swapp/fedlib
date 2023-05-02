import logging
import json
import os

__all__ = ['get_logger', 'init_logs','mkdirs']

def get_logger():
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger

def init_logs(log_file_name, args=None, log_dir=None):
    
    #mkdirs(log_dir)
    try:
        os.makedirs(os.path.dirname(log_dir))
    except FileExistsError:
        # directory already exists, do nothing
        pass


    #mkdirs(os.path.dirname(log_dir), exist_ok=True)

    argument_path = log_file_name + '-arguments.json'
    log_path = log_file_name + '-results.log'


    #TODO change to yaml
    with open(os.path.join(log_dir, argument_path), 'w') as f:
        json.dump(str(args), f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)


    logging.basicConfig(
        filename=os.path.join(log_dir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    
    return logger

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
