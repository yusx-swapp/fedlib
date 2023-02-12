import configparser
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "--cf", help="Path to the configuration file")
    args = parser.parse_args()
    
    if not args:
        raise FileNotFoundError("No configuration file provided or the file is empty"))

    config = configparser.ConfigParser()
    config.read(args.config)
    
    return config

if __name__ == "__main__":
    args = parse_args()
    print(args)
