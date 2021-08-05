import sys
import os
import yaml

from FL_basic import start_fl


global base_path
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    global base_path
    config_file = base_path + '/configs/' + sys.argv[1]
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)
        ## don't modify config here, just update the setting in corresponding modified config file and give the file name
        start_fl(config)


if __name__ == "__main__":
    main()
