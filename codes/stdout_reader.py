## Read output from output and create a csv file
## Also check which datasets didn't run the corresponding jobs


import os
import pandas as pd
import glob
import pprint
import json

base_path = os.path.dirname(os.path.realpath(__file__)).split('codes')[0]
pp = pprint.PrettyPrinter(indent=4)

if __name__ == '__main__':
    # list all the slurm outputs
    # datasets = pd.read_csv(os.path.join(base_path, 'data','dataset_details.csv'))
    data_exp = {}

    results = glob.glob('*.log')
    print("Found {} logs".format(len(results)))
    modes = ['train','test','val']
    rows = []
    for res in results:
        r_file = open(res).readlines()
        for rf in r_file:
            ep_max_epoch = 0
            data = ''
            experiment_name = ''
            if "togrep_" in rf:
                sp = rf.split(' ; ')
                mode = sp[0].rstrip().split('togrep_')[-1]
                epoch = int(sp[2].split(' : ')[-1].lstrip().rstrip())
                ep_max_epoch = max(ep_max_epoch, epoch)
                data = sp[3].split(' : ')[-1]
                experiment_name = sp[1]
                rows.append(
                    {'mode' : mode,
                     'experiment_name': experiment_name,
                     'epoch': epoch,
                     'data' : data,
                     'file' : sp[4].split(' : ')[-1],
                     'loss' : sp[5].split(' : ')[-1],
                     'accuracy' : sp[6].split(' : ')[-1].rstrip()
                     })
            if data not in data_exp:
                data_exp[data] = {}
            if experiment_name not in data_exp[data]:
                data_exp[data][experiment_name] = 0
            data_exp[data][experiment_name] = ep_max_epoch
    df = pd.DataFrame(rows)
    df.to_csv('all_results_runs.csv')
    pp.pprint(data_exp)
    json.dump(data_exp, open('run_metrics.json','w'))
