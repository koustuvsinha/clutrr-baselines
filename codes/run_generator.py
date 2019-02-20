# generate run scripts
import argparse
import glob
import os
import random
import yaml
import json

base_path = os.path.dirname(os.path.realpath(__file__)).split('codes')[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--comet_api", type=str, default='')
    parser.add_argument("--comet_workspace", type=str, default='***REMOVED***')
    parser.add_argument("--comet_project", type=str, default='compositionality-nli')
    parser.add_argument('--loc', type=str, default='data/')
    parser.add_argument('--models', type=str, default='bilstm,gat_clean,lstm_atten,mac,rn,rn_tpr')
    parser.add_argument('--local', action='store_true', help="If true, run on machines not on slurm")
    parser.add_argument('--gpus',type=str, default='0', help='works in local, run jobs on these gpus')
    parser.add_argument('--stdout', type=str, default='std_outputs', help='folder to store std outputs')

    args = parser.parse_args()

    models = args.models.split(',')
    gpus = args.gpus.split(',')
    if not os.path.exists(args.stdout):
        os.makedirs(args.stdout)
    if not args.local and len(gpus) > 1:
        print("set gpus to 0 for non local jobs")
        exit(0)
    folders = glob.glob(os.path.join(base_path, args.loc) + '*/')
    print("Found {} folders".format(len(folders)))
    ct = 0
    run_flnames = {gpu:[] for gpu in gpus}
    for folder in folders:
        print("Directory : {}".format(folder))
        base_data_name = folder.split('/')[-2]
        print("Data name",base_data_name)
        data_config = json.load(open(os.path.join(folder, 'config.json'),'r'))
        train_task = ','.join(data_config['train_task'].keys())
        test_task = ','.join(data_config['test_tasks'].keys())
        f_key = list(data_config['args'].keys())[0]
        holdout = data_config['args'][f_key]['holdout'] if 'holdout' in data_config['args'][f_key] else ''
        data_desc = 'train_{}.test_{}.holdout_{}'.format(train_task, test_task, holdout)
        gpu_choice = random.choice(gpus)
        run_file = "#!/bin/sh\n"
        if not args.local:
            run_file += "#SBATCH --time=0-0:30\n"
            run_file += "#SBATCH --account=rrg-dprecup\n"
            run_file += "#SBATCH --ntasks=16\n"
            run_file += "#SBATCH --gres=gpu:1\n"
            run_file += "#SBATCH --mem=0\n"
            run_file += "module load nixpkgs/16.09\n"
            run_file += "module load intel/2018.3\n"
            run_file += "module load cuda/10.0.130\n"
            run_file += "module load cudnn/7.4\n"
            run_file += "source activate compnli\n"
            run_file += "cd /home/***REMOVED***/projects/def-jpineau/***REMOVED***/InferSent-comp/\n"
        run_file += "export COMET_API='{}'\n".format(args.comet_api)
        run_file += "export COMET_WORKSPACE='{}'\n".format(args.comet_workspace)
        run_file += "export COMET_PROJECT='{}'\n".format(args.comet_project)
        run_file += "export PYTHONPATH={}\n".format(base_path)
        if args.local:
            run_file += "cd {}codes/app/\n".format(base_path)
        run_file += "\n"

        for model in models:
            if args.local:
                pre = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_choice)
            else:
                pre = ''
            output_file = '{}/{}_{}.out'.format(args.stdout, base_path, model)
            config_file = yaml.load(open(os.path.join(base_path, 'config', '{}.yaml'.format(model))))
            config_file['dataset']['data_path'] = base_data_name
            config_file['dataset']['data_desc'] = data_desc
            yaml.dump(config_file, open(os.path.join(base_path, 'config', '{}_{}.yaml'.format(model, base_data_name)),'w'))
            run_file += pre + "python main.py --config_id {}_{}\n".format(model, base_data_name)
            ct += 1
        run_file += "\n"

        print("Writing file")
        if not os.path.exists("runs"):
            os.makedirs("runs")
        last_path = folder.split('/')[-2]
        run_flname = 'runs/run_{}.sh'.format(last_path)
        with open(run_flname, 'w') as fp:
            fp.write(run_file)
        run_flnames[gpu_choice].append(run_flname)
    print("Done, now writing the meta runner")
    for gpu in gpus:
        meta_file = "#!/bin/sh\n"
        for rf in run_flnames[gpu]:
            if not args.local:
                meta_file += "sbatch {}\n".format(rf)
            else:
                meta_file += "./{}\n".format(rf)
        with open('meta_run_{}.sh'.format(gpu),'w') as fp:
            fp.write(meta_file)
    print("Number of experiments to run : {}".format(ct))





