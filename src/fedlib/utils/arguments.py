import configparser
import argparse

__all__ = ['Arguments']

class Arguments:
    
    class __args:
        pass
    
    def __init__(self, config_file=None):
        self.config = self.parse_args(config_file)
        
        for section in self.config.sections():
            setattr(self, section, Arguments.__args())
            for key, value in self.config.items(section):
                if not value:
                    pass
                elif value.lower() in ['true','t']:
                    value = True
                elif value.lower() in ['false', 'f']:
                    value = False
                else:
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                setattr(getattr(self, section), key, value)

    def parse_args(self,config_file):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", "--cf", help="Path to the configuration file")
        args = parser.parse_args()
        
        if config_file is not None:
            args.config=config_file
        
        if not args.config:
            raise FileNotFoundError("No configuration file provided or the file is empty")

        config = configparser.ConfigParser()
        config._interpolation = configparser.ExtendedInterpolation()

        config.read(args.config)
        
        return config

    def show_args(self):
        for section in self.config.sections():
            category = getattr(self, section)
            for key, value in vars(category).items():
                print(f"{section}.{key} = {value}")
