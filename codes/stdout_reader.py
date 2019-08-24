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
    data_prob = {}
    all_datasets = glob.glob(os.path.join(base_path, 'data', 'data*'))
    all_datasets = set([dir.split('/')[-1] for dir in all_datasets if os.path.isdir(dir)])

    results = glob.glob('*.err')
    print("Found {} logs".format(len(results)))
    modes = ['train','test','val']
    rows = []
    for res in results:
        test_files = set()
        r_file = open(res).readlines()
        ep_max_epoch = 0
        data = ''
        experiment_name = ''
        for rf in r_file:
            if "togrep_" in rf:
                sp = rf.split(' ; ')
                mode = sp[0].rstrip().split('togrep_')[-1]
                if mode == 'final':
                    continue
                epoch = int(sp[2].split(' : ')[-1].lstrip().rstrip())
                ep_max_epoch = max(ep_max_epoch, epoch)
                data = sp[3].split(' : ')[-1]
                experiment_name = sp[1]
                run_file = sp[4].split(' : ')[-1].lstrip()
                if ';;' in sp[5]:
                    sp_ = sp[5].split(' ;; ')
                    sp[5] = sp_[0]
                    sp.append(sp_[-1])
                rows.append(
                    {'mode' : mode,
                     'experiment_name': experiment_name,
                     'epoch': epoch,
                     'data' : data,
                     'file' : run_file,
                     'loss' : sp[5].split(' : ')[-1],
                     'accuracy' : sp[6].split(' : ')[-1].rstrip()
                     })
                if mode == 'test':
                    test_files.add(run_file.split('/')[-1])
        if data in all_datasets:
            if data not in data_exp:
                data_exp[data] = {}
            if experiment_name not in data_exp[data]:
                data_exp[data][experiment_name] = 0
            data_exp[data][experiment_name] = ep_max_epoch
            # check for running of all test files
            data_config = json.load(open(os.path.join(base_path, 'data', data, 'config.json')))
            test_fls = [tf for tt,tf in data_config['test_tasks'].items()]
            done_fls = set(test_fls).intersection(test_files)
            #print(test_fls)
            #print(test_files)
            #exit(0)
            if len(done_fls) != len(test_fls):
                if data not in data_prob:
                    data_prob[data] = []
                data_prob[data].append(experiment_name)
                print("experiment {} with data {} didn't run all tests".format(experiment_name, data))
    df = pd.DataFrame(rows)
    df.to_csv('all_results_runs.csv')
    not_complete = list(all_datasets - set(data_exp.keys()))
    data_exp['Not Complete'] = not_complete
    pp.pprint(data_exp)
    pp.pprint(data_prob)
    print("Total datasets : {}, Datasets ran on : {}".format(len(all_datasets),len(data_exp)))
    print("Not complete : {}".format(not_complete))
    json.dump(data_exp, open('run_metrics.json','w'))

