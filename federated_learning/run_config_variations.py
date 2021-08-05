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
        
        ## any type of variations can be added in nested structure
        ## change POISON_FRAC
        ## change CLIENT_FRAC
        for posion_frac in [0.0, 0.1, 0.2, 0.4]:
            config['CREATE_DATASET'] = True
            config['POISON_FRAC'] = posion_frac
            for client_frac in [0.1, 0.2, 0.4, 1.0]:
                config['CLIENT_FRAC'] = client_frac
                start_fl(config)




if __name__ == "__main__":
    main()
