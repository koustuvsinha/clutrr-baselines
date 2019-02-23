## Read output from output and create a csv file
## Check which iteration has the best config


import pandas as pd
import glob

if __name__ == '__main__':
    # list all the slurm outputs
    results = glob.glob('*.log')
    print("Found {} logs".format(len(results)))
    rows = []
    for res in results:
        r_file = open(res).readlines()
        for rf in r_file:
            if "> togrep :" in rf:
                sp = rf.split(' : ')
                m_test_acc = float(sp[-1].lstrip())
                file_name = sp[1]
                model_name = file_name.split('_hp')[0]
                rows.append({'file': file_name, 'model_name': model_name, 'test_acc':m_test_acc})
    df = pd.DataFrame(rows)
    df.to_csv('hyp_results.csv')
    print("Best hyperparams : ")
    print(df.sort_values('test_acc', ascending=False).drop_duplicates(['model_name']))
