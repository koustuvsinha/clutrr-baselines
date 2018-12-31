import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description="Argument parser to obtain the name of the config file")
    parser.add_argument('--config_id', default="graph_rel", help='config id to use')
    parser.add_argument('--exp_id', default='', help='optional experiment id to resume')
    args = parser.parse_args()
    return args.config_id, args.exp_id