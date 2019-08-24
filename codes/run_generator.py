# generate run scripts
import argparse
import glob
import os
import random
import yaml
import json

base_path = os.path.dirname(os.path.realpath(__file__)).split('codes')[0]

def run_per_folder(args, run_num=0):
    models = args.models.split(',')
    gpus = args.gpus.split(',')
    local_gpu_jobs = {gpu: [] for gpu in gpus}
    if not os.path.exists(args.stdout):
        os.makedirs(args.stdout)
    if not args.local and len(gpus) > 1:
        print("set gpus to 0 for non local jobs")
        exit(0)
    folders = glob.glob(os.path.join(base_path, args.loc) + '*/')
    print("Found {} folders".format(len(folders)))
    ct = 0
    run_flnames = []

    # file paths
    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    config_dir = os.path.join(path, "config")
    script_subf = 'local' if args.local else 'slurm'
    script_dir = os.path.join(path, "scripts", script_subf)
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)

    for folder in folders:
        print("Directory : {}".format(folder))
        base_data_name = folder.split('/')[-2]
        print("Data name", base_data_name)
        if len(args.only_data) > 0:
            if args.only_data not in folder:
                continue
        models_str = '_'.join(models)
        exp_str = '{}_{}_{}'.format(base_data_name, models_str, run_num)
        wrapper_file_name = 'wrapper_{}.sh'.format(exp_str)
        data_config = json.load(open(os.path.join(folder, 'config.json'), 'r'))
        train_task = ','.join(data_config['train_task'].keys())
        test_task = ','.join(data_config['test_tasks'].keys())
        f_key = list(data_config['args'].keys())[0]
        holdout = data_config['args'][f_key]['holdout'] if 'holdout' in data_config['args'][f_key] else ''
        data_desc = 'train_{}.test_{}.holdout_{}'.format(train_task, test_task, holdout)
        run_file = "#!/bin/sh\n"
        # change
        if not args.local:
            run_file += "#SBATCH --job-name=clutrr_{}\n".format(exp_str)
            run_file += "#SBATCH --output=/checkpoint/koustuvs/jobs/{}_%j.out\n".format(exp_str)
            run_file += "#SBATCH --error=/checkpoint/koustuvs/jobs/{}_%j.err\n".format(exp_str)
            run_file += "#SBATCH --comment=\"EMNLP Deadline 21/05/19\"\n"
            run_file += "#SBATCH --partition={}\n".format(args.partition)
            run_file += "#SBATCH --nodes=1\n"
            run_file += "#SBATCH --ntasks-per-node={}\n".format(len(models))
            run_file += "#SBATCH --gres=gpu:{}\n".format(len(models))
            run_file += "#SBATCH --cpus-per-task 2\n"
            run_file += "#SBATCH --time 6:00:00\n"
            run_file += "module purge\n"
            run_file += "module load openmpi/3.0.0/gcc.5.4.0\n"
            run_file += "module load NCCL/2.4.2-1-cuda.10.0\n"
            run_file += "module load cuda/10.0\n"
            run_file += "module load cudnn/v7.4-cuda.10.0\n"
            # run_file += "module load anaconda3\n"
            run_file += "\n"

            # end of batch file, which should now run the corresponding wrapper
            run_file += "srun --label {}".format(wrapper_file_name)
            print("Writing sbatch file")
            run_flname = 'run_{}.sh'.format(exp_str)
            with open(os.path.join(script_dir, run_flname), 'w') as fp:
                fp.write(run_file)
            run_flnames.append(run_flname)
            wrapper_file = "#!/bin/sh\n"
            # wrapper_file += "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID\n"
            wrapper_file += "echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES\n"
            wrapper_file += "./{}_model_$SLURM_LOCALID.sh\n".format(exp_str)
            wrapper_file += "\n"
            with open(os.path.join(script_dir, wrapper_file_name), 'w') as fp:
                fp.write(wrapper_file)

        for mid, model in enumerate(models):
            gpu_choice = random.choice(gpus)
            if args.local:
                pre = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_choice)
            else:
                pre = ''
            model_run_file = "#!/bin/sh\n"
            output_file = '{}/{}_{}.out'.format(args.stdout, base_path, model)
            # model_run_file += ". /private/home/koustuvs/miniconda3/bin/activate gnnlogic\n"
            model_run_fl_name = os.path.join(script_dir,
                                             '{}_model_{}.sh'.format(exp_str, mid))
            model_run_file += "export COMET_API='{}'\n".format(args.comet_api)
            model_run_file += "export COMET_WORKSPACE='{}'\n".format(args.comet_workspace)
            model_run_file += "export COMET_PROJECT='{}'\n".format(args.comet_project)
            model_run_file += "export PYTHONPATH=$PYTHONPATH:{}\n".format(path)
            model_run_file += "export PATH=/private/home/koustuvs/miniconda3/envs/gnnlogic/bin/:$PATH\n"
            model_run_file += "which python\n"
            model_run_file += "echo 'Choosing GPU'\n"
            model_run_file += "timestamp() {\n"
            model_run_file += "  date +\"%Y-%m-%d_%H-%M-%S\"\n"
            model_run_file += "}\n"
            model_run_file += "cd {}/codes/app\n".format(base_path)
            if args.local:
                model_run_file += "echo \"$(timestamp) Start running {}\"\n".format(model_run_fl_name)
            run_path = os.path.join(path, 'codes', 'app')
            checkpoint_loc = '/checkpoint/koustuvs/clutrr/std_outputs/{}.out'.format(exp_str)
            ent_policies = args.entity_policy.split(',')
            for ep in ent_policies:
                config_file = yaml.load(open(os.path.join(base_path, 'config', '{}.yaml'.format(model))))
                config_file['dataset']['data_path'] = base_data_name
                config_file['dataset']['data_desc'] = data_desc
                config_file['general']['base_path'] = '/checkpoint/koustuvs/clutrr/'
                config_file['model']['embedding']['entity_embedding_policy'] = ep
                config_file['model']['num_epochs'] = args.num_epochs
                config_file['model']['embedding']['dim'] = args.emb_dim
                config_file['model']['optimiser']['name'] = args.optim
                config_file['model']['optimiser']['learning_rate'] = args.lr
                config_file['model']['optimiser']['clip'] = args.clip
                if args.runs > 1:
                    # set a different seed for each run
                    config_file['general']['seed'] = config_file['general']['seed'] + run_num
                config_file_name = '{}_{}_{}_{}'.format(model, base_data_name, ep, run_num)
                yaml.dump(config_file,
                          open(os.path.join(base_path, 'config', '{}.yaml'.format(config_file_name)), 'w'),
                          default_flow_style=False)
                model_run_file += pre + "python {}/main.py --config_id {} > {}\n".format(run_path, config_file_name,
                                                                                        checkpoint_loc)
            model_run_file += 'echo "$(timestamp) Done running"'
            with open(model_run_fl_name, 'w') as fp:
                fp.write(model_run_file)
            local_gpu_jobs[gpu_choice].append(model_run_fl_name)
            ct += 1
    print("Done, now writing the meta runner")
    mt = 0
    meta_run_files = []
    for gpu in gpus:
        meta_file = "#!/bin/sh\n"
        if not args.local:
            for rf in run_flnames:
                meta_run_files.append("sbatch {}\n".format(rf))
                mt += 1
        else:
            for mfile in local_gpu_jobs[gpu]:
                meta_run_files.append(". {}\n".format(mfile))
                mt += 1
    return ct, mt, meta_run_files



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--comet_api", type=str, default='')
    parser.add_argument("--comet_workspace", type=str, default='koustuvs')
    parser.add_argument("--comet_project", type=str, default='compositionality-nli')
    parser.add_argument('--loc', type=str, default='data/')
    parser.add_argument('--models', type=str, default='bilstm_atten,bilstm_max,bilstm_mean,bilstm_concat,gat_clean,mac,rn,rn_tpr')
    parser.add_argument('--local', action='store_true', help="If true, run on machines not on slurm")
    parser.add_argument('--cluster', type=str, default='graham', help="graham/fair")
    parser.add_argument('--gpus',type=str, default='0', help='works in local, run jobs on these gpus')
    parser.add_argument('--stdout', type=str, default='std_outputs', help='folder to store std outputs')
    parser.add_argument('--only_data', type=str, default='', help='if this is set, use only the said dataset')
    parser.add_argument('--runs', type=int, default=1, help='number of runs to perform')
    parser.add_argument('--entity_policy', type=str, default='fixed', help='can be fixed/random/learned separated by comma')
    parser.add_argument('--num_epochs', type=int, default=50, help='num epochs')
    parser.add_argument('--emb_dim', type=int, default=100, help='emb_dim')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--clip', type=int, default=8, help='clipping (0 for no clipping)')
    parser.add_argument('--partition', type=str, default='learnfair', help='learnfair/dev/uninterrupted')

    args = parser.parse_args()

    ct = 0
    mt = 0
    mfiles = []
    for i in range(args.runs):
        cti, mti, mf = run_per_folder(args, i)
        ct += cti
        mti += mt
        mfiles.extend(mf)

    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    script_subf = 'local' if args.local else 'slurm'
    script_dir = os.path.join(path, "scripts", script_subf)
    with open(os.path.join(script_dir, 'meta_run.sh'), 'a') as fp:
        fp.writelines(mfiles)








