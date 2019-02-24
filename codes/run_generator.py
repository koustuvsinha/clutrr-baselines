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
    parser.add_argument('--models', type=str, default='bilstm_atten,bilstm_max,bilstm_mean,bilstm_concat,gat_clean,mac,rn,rn_tpr')
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
    run_flnames = []

    # file paths
    path = os.path.dirname(os.path.realpath(__file__)).split('/codes')[0]
    config_dir = os.path.join(path, "config")
    script_dir = os.path.join(path, "scripts")
    if not os.path.exists(script_dir):
        os.makedirs(script_dir)

    for folder in folders:
        print("Directory : {}".format(folder))
        base_data_name = folder.split('/')[-2]
        print("Data name",base_data_name)
        wrapper_file_name = 'wrapper_{}_{}.sh'.format(base_data_name, '_'.join(models))
        data_config = json.load(open(os.path.join(folder, 'config.json'),'r'))
        train_task = ','.join(data_config['train_task'].keys())
        test_task = ','.join(data_config['test_tasks'].keys())
        f_key = list(data_config['args'].keys())[0]
        holdout = data_config['args'][f_key]['holdout'] if 'holdout' in data_config['args'][f_key] else ''
        data_desc = 'train_{}.test_{}.holdout_{}'.format(train_task, test_task, holdout)
        gpu_choice = random.choice(gpus)
        run_file = "#!/bin/sh\n"
        if not args.local:
            run_file += "#SBATCH --job-name=clutrr_{}_{}\n".format(base_data_name, '_'.join(models))
            run_file += "#SBATCH --output=/checkpoint/***REMOVED***/jobs/{}_{}_%j.out\n".format(base_data_name, '_'.join(models))
            run_file += "#SBATCH --error=/checkpoint/***REMOVED***/jobs/{}_{}_%j.err\n".format(base_data_name, '_'.join(models))
            run_file += "#SBATCH --partition=uninterrupted\n"
            run_file += "#SBATCH --nodes=1\n"
            run_file += "#SBATCH --ntasks-per-node={}\n".format(len(models))
            run_file += "#SBATCH --gres=gpu:{}\n".format(len(models))
            run_file += "#SBATCH --cpus-per-task 4\n"
            run_file += "#SBATCH --time 06:00:00\n"
            run_file += "module purge\n"
            run_file += "module load openmpi/3.0.0/gcc.5.4.0\n"
            run_file += "module load NCCL/2.4.2-1-cuda.10.0\n"
            run_file += "module load cuda/10.0\n"
            run_file += "module load cudnn/v7.4-cuda.10.0\n"
            #run_file += "module load anaconda3\n"
        run_file += ". /private/home/***REMOVED***/miniconda3/bin/activate gnnlogic\n"
        
        if args.local:
            run_file += "cd {}codes/app/\n".format(base_path)
        run_file += "\n"

        # end of batch file, which should now run the corresponding wrapper
        run_file += "srun --label {}".format(wrapper_file_name)
        print("Writing sbatch file")
        run_flname = 'run_{}_{}.sh'.format(base_data_name, '_'.join(models))
        with open(os.path.join(script_dir, run_flname), 'w') as fp:
            fp.write(run_file)
        run_flnames.append(run_flname)
        wrapper_file = "#!/bin/sh\n"
        #wrapper_file += "export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID\n"
        wrapper_file += "echo $SLURMD_NODENAME $SLURM_JOB_ID $CUDA_VISIBLE_DEVICES\n"
        wrapper_file += "./{}_{}_model_$SLURM_LOCALID.sh\n".format(base_data_name, '_'.join(models))
        wrapper_file += "\n"
        with open(os.path.join(script_dir, wrapper_file_name), 'w') as fp:
            fp.write(wrapper_file)

        for mid, model in enumerate(models):
            if args.local:
                pre = 'CUDA_VISIBLE_DEVICES={} '.format(gpu_choice)
            else:
                pre = ''
            model_run_file = "#!/bin/sh\n"
            output_file = '{}/{}_{}.out'.format(args.stdout, base_path, model)
            config_file = yaml.load(open(os.path.join(base_path, 'config', '{}.yaml'.format(model))))
            config_file['dataset']['data_path'] = base_data_name
            config_file['dataset']['data_desc'] = data_desc
            config_file['general']['base_path'] = '/checkpoint/***REMOVED***/clutrr/'
            yaml.dump(config_file, open(os.path.join(base_path, 'config', '{}_{}.yaml'.format(model, base_data_name)),'w'), default_flow_style=False)
            # model_run_file += ". /private/home/***REMOVED***/miniconda3/bin/activate gnnlogic\n"
            model_run_file += "export COMET_API='{}'\n".format(args.comet_api)
            model_run_file += "export COMET_WORKSPACE='{}'\n".format(args.comet_workspace)
            model_run_file += "export COMET_PROJECT='{}'\n".format(args.comet_project)
            model_run_file += "export PYTHONPATH=$PYTHONPATH:{}\n".format(path)
            model_run_file += "export PATH=/private/home/***REMOVED***/miniconda3/envs/gnnlogic/bin/:$PATH\n"
            model_run_file += "which python\n"
            model_run_file +="echo 'Choosing GPU : $CUDA_VISIBLE_DEVICES'\n"
            run_path = os.path.join(path, 'codes','app')
            checkpoint_loc = '/checkpoint/***REMOVED***/clutrr/std_outputs/{}_{}.out'.format(model, base_data_name)
            model_run_file += pre + "python {}/main.py --config_id {}_{} > {}\n".format(run_path, model, base_data_name, checkpoint_loc)
            with open(os.path.join(script_dir, '{}_{}_model_{}.sh'.format(base_data_name, '_'.join(models), mid)),'w') as fp:
                fp.write(model_run_file)
            ct += 1
    print("Done, now writing the meta runner")
    mt = 0
    for gpu in gpus:
        meta_file = "#!/bin/sh\n"
        for rf in run_flnames:
            if not args.local:
                meta_file += "sbatch {}\n".format(rf)
            else:
                meta_file += "./{}\n".format(rf)
            mt +=1
        with open(os.path.join(script_dir, 'meta_run_{}.sh'.format(gpu)),'w') as fp:
            fp.write(meta_file)
    print("Number of experiments to run : {}".format(ct))
    print("Number of batches to submit : {}".format(mt))





