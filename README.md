# Experiment Base

Base folder which should be forked to start any experiment. Contains all basic boilerplate and management.

## Folder organization

- **codes** - contains all the code required for the experiment
    - **app** - starting point of the experiment, using `main.py`. In general to start an experiment, use `python main.py config_id <experiment_name>`.
    - **baselines** - folder to contain all the baselines. Basic LSTM baseline is implemented in this repo.
    - **experiment** - Experiment running and maintainance code is in `experiment.py`. 
    - **metric** - contains metrics to calculate and track
    - **models** - should contain the model which is being worked on this experiment. In this base, it contains the basic GNN model we used for the paper.
    - **net** - contains modelling boilerplates, such as base network, attention modules, generators, and batch classes.
    - **utils** - utility belt containing various reusable utilities for the entire experiment
- **config** - contains hyperparams and configs in `yaml` files.
- **logs** - contains logs in csv, although it is redundant at this point since we use Comet. Maybe we need to look into how to export / save logs from Comet for publication
- **plots** - should be used to create plots and visualizations for the experiments
- **docs** - may contain useful docs regarding the model

_Generated folders_

- **model** - contains checkpoints for all the experiments

## Modelling concepts

- Any model should have two subclasses - **encoder** and **decoder**
- Each class should inherit `codes.net.base_net.Net`
- All models are run by an orchestrator `codes.net.trainer.Trainer`
- Therefore, all model code / attention / etc should remain only within the `encoder` and `decoder`.
- `Trainer` supports two modes of training : `classify` for n-way classification tasks, and `seq2seq` for language generation tasks.
- All metric calculation and tracking is done in `codes.metric` package. This should also incorporate
early stopping criterias.

## Config Management

- Hyperaparam configs are stored in `config/` folder in `yaml` files.
- `yaml` files provide the option of using comments while maintaining a dictionary
- To run a new experiment, copy the `sample.config.yaml` into `<experiment_name>.config.yaml`
- Change/Modify configs in this new file. If you need to add new fields, add them both in this file and `sample.config.yaml` with comments
- When running an experiment, provide the name of the config file: `python main.py --config_id <experiment_name>`


## Experiment Management

- We use [Comet.ml](https://comet.ml) to monitor and log experiments / accuracies
- After creating an account in [Comet.ml](https://comet.ml), (Acedmic accounts are free) put the key and workspace info in `config.log`
- To debug, use `config.log.disabled=True` to not pollute the Comet dashboard space.
- Each experiment has an unique hex experiment_id, which is shown in Comet dashboard
- Each experiment is _resumable_. The checkpoints are created in `model/` folder with the experiment name. To resume an experiment,
put the argument `--exp_id` along with `--config_id` in `main.py`, where
the `exp_id` should be the hex.



