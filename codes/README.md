## Organization

### Folder organization

- **codes** - contains all the code required for the experiment
    - **app** - starting point of the experiment, using `main.py`. In general to start an experiment, use `python main.py config_id <experiment_name>`.
    - **baselines** - folder to contain all the baselines.
    - **experiment** - Experiment running and maintainance code is in `experiment.py`. 
    - **metric** - contains metrics to calculate and track
    - **net** - contains modelling boilerplates, such as base network, attention modules, generators, and batch classes.
    - **utils** - utility belt containing various reusable utilities for the entire experiment
- **config** - contains hyperparams and configs in `yaml` files.
- **logs** - contains logs in csv, although it is redundant at this point since we use Comet. Maybe we need to look into how to export / save logs from Comet for publication
- **plots** - should be used to create plots and visualizations for the experiments
- **docs** - may contain useful docs regarding the model

_Generated folders_

- **model** - contains checkpoints for all the experiments

### Modelling concepts

- Any model should have two subclasses - **encoder** and **decoder**
- Each class should inherit `codes.net.base_net.Net`
- All models are run by an orchestrator `codes.net.trainer.Trainer`
- Therefore, all model code / attention / etc should remain only within the `encoder` and `decoder`.
- `Trainer` supports two modes of training : `classify` for n-way classification tasks, and `seq2seq` for language generation tasks.
- All metric calculation and tracking is done in `codes.metric` package. This should also incorporate
early stopping criterias.
- To specify which encoder and decoder to run for the current experiment, modify `config.model.encoder.name` and  `config.model.decoder.name` for dynamic loading.

### Config Management

- Hyperaparam configs are stored in `config/` folder in `yaml` files.
- `yaml` files provide the option of using comments while maintaining a dictionary
- To run a new experiment, copy the `sample.config.yaml` into `<experiment_name>.config.yaml`
- Change/Modify configs in this new file. If you need to add new fields, add them both in this file and `sample.config.yaml` with comments
- When running an experiment, provide the name of the config file: `python main.py --config_id <experiment_name>`
