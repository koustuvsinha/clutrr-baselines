# CLUTRR-Baselines

Codebase for experiments on the [CLUTRR benchmark suite](https://github.com/facebookresearch/clutrr/).

- Paper: https://arxiv.org/abs/1908.06177
- Webpage: https://www.cs.mcgill.ca/~ksinha4/introducing-clutrr/

### Usage

- We use [Comet.ml](https://comet.ml) to monitor and log experiments / accuracies
- After creating an account in [Comet.ml](https://comet.ml), (Academic accounts are free) put the key and workspace info in `config.log`
- To debug, use `config.log.disabled=True` to not pollute the Comet dashboard space.
- To run an experiment, first make sure that the codebase is in the Pythonpath:
`export PYTHONPATH="${PYTHONPATH}:/home/.../experiment_base`
- Then, cd into `codes/app` and run `python main.py --config_id <experiment_name>`
- `experiment_name` is the name of the config file (without .yaml)

### Dependencies

- Pytorch, 1.0
- Comet.ml
- Addict
- NLTK
- Pandas
- Pyyaml
- tqdm
