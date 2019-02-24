## Read output from output and create a csv file
## Check which iteration has the best config


import pandas as pd
import glob

if __name__ == '__main__':
    # list all the slurm outputs
    results = glob.glob('*.log')
    print("Found {} logs".format(len(results)))
    modes = ['train','test','val']
    rows = []
    for res in results:
        r_file = open(res).readlines()
        for rf in r_file:
            if "togrep_" in rf:
                sp = rf.split(' ; ')
                mode = sp[0].rstrip().split('togrep_')[-1]
                rows.append(
                    {'mode' : mode,
                     'experiment_name': sp[1],
                     'epoch': sp[2].split(' : ')[-1],
                     'data' : sp[3].split(' : ')[-1],
                     'file' : sp[4].split(' : ')[-1],
                     'loss' : sp[5].split(' : ')[-1],
                     'accuracy' : sp[6].split(' : ')[-1].rstrip()
                     })
    df = pd.DataFrame(rows)
    df.to_csv('all_results.csv')
