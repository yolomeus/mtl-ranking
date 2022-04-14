# Multi-Task Learning for Ranking

Project investigating multi-task learning for ranking, using pytorch and pytorch-lightning.

## Requirements

All packages required for this project are defined in `environment.yml` and can be installed with
[anaconda](https://www.anaconda.com/). After installing anaconda, simply run

```shell
conda env create -f environment.yml
```

from the repository root, to create a virtual conda environment with the correct dependencies installed. Then activate
the environment like so:

```shell
conda activate mtl-ranking
```

## Quickstart: Running an Experiment

The project uses [hydra](https://hydra.cc/) for configuration. For details on configuration
see [Project Structure](#project-structure).

To run an experiment, use the `train.py` script. You can change experiment settings by passing command line arguments or
by editing the config files found in `conf/`.

```shell
python train.py training.batch_size=32 loop.optimizer.lr=1e-4
```

## Project Structure

For structuring this project we use a composable hydra configuration (found in `conf/`) which mirrors the structure and
composition of the python modules we use. This enables us to hot-swap parts of our training-pipeline and set up
different experiments without directly editing the code.

```
conf
│
├───config.yaml # the root config file
│
├───datamodule
│   ├───dataset
│   │   ├───metrics
│   │   └───preprocessor
│   └───sampler
├───loop
│   ├───loss
│   └───optimizer
└───model
    ├───body
    └───heads
        └───pooler
```

For example, if we want to add a new dataset, all we need to do is add a `my_dataset.yaml` config file
to `conf/datamodule/dataset/` with a `_target_` field that points to our dataset python class, e.g.
`_target_: datamodule.dataset.MyDataset` and select it when running the train script:

```shell
python train.py datamodule/dataset=my_dataset
```

The following explains each module, its role in the pipeline and how to replace it.

### Model

The pytorch model we train. A MultiTaskModel consist of a model body and a head for each dataset. A heads name needs to
match the name of a dataset to be properly mapped to samples coming from that dataset.

```yaml
# conf/model/multitask_model.yaml
defaults:
  - body: bert_base_uncased
  # select msm_bm25 from the heads/ config group and place it at .heads.msm_bm25 in this config
  - heads@heads.msm_bm25: msm_bm25
  - heads@heads.trec2019: trec2019

_target_: model.multitask.MultiTaskModel
```

```python
class MultiTaskModel(Module):
    def __init__(self, body: Module, heads: Dict[str, Module]):
        ...
```

A head module should implement `model.head.MTLHead` and a body `model.body.MTLBody`.

### Loop

A pytorch-lightning module that defines train, validation and test steps. It also holds the optimizer, loss and metric
modules as well as the model.

```python
class MultiTaskLoop(LightningModule):
    def __init__(self, hparams: DictConfig, model: MultiTaskModel, optimizer: Optimizer, loss: Module):
        ...
```

### Datamodule

The datamodule provides dataloaders and defines how to pre-process and assemble each dataset. For this project we use an
MTL specific datamodule that holds a reference to a dataset and dataloader samplers for {train, validation, test}_split.
We most likely do not need to replace the full datamodule for MTL experiments.

```yaml
defaults:
  - dataset: mtl
  - sampler@train_sampler: random_proportional
  - sampler@val_sampler: sequential
  - sampler@test_sampler: sequential

_target_: datamodule.multitask_datamodule.MTLDataModule
...
```

#### Dataset

The dataset encapsulated by the datamodule. For the MTL datamodule, we also use a general MTL dataset
(`conf/datamodule/dataset/mtl.yaml`) which encapsulates multiple datasets. We can add a dataset by implementing it in
`datamodule.dataset` and defining its config in `conf/datamodule/dataset`.   
Then list its name in the `datasets` property of `mtl.yaml`:

```yaml 
# conf/datamodule/dataset/mtl.yaml
defaults:
  - ../dataset@datasets:
      # select the msm_bm25 dataset from ../dataset and pass it as 
      # MTLDataset's datasets constructor arg  
      - msm_bm25
      - trec2019
_target_: datamodule.dataset.MTLDataset
```

```python
class MTLDataset(PreparedDataset):
    def __init__(self, datasets: Dict[str, PreparedDataset]):
        ...
```

To work with MTLDataset, a dataset should subclass `datamodule.dataset.PreparedDataset`.

We can override parameters passed to the dataset like so:

```shell
python train.py datamodule.dataset.datasets.msm_bm25.name="msm_bm25_new"
```

#### Sampler

The sampler defines how to sample from each dataset by returning indexes and has to subclass `
torch.utils.data.Sampler`. For the MTL dataset, instead of a single
integer index it has to return a tuple `(dataset_idx, sample_idx)`. This way, it specifies not only a sample's
position in a dataset, but also from which dataset to sample from.
We need to provide each {train, val, test}_sampler_cfg to the datamodule:

```yaml
# conf/datamodule/multitask.yaml
defaults:
  - dataset: mtl
  # select random_proportional from ./sampler and set as
  # datamodule.train_sampler
  - sampler@train_sampler_cfg: random_proportional
  - sampler@val_sampler_cfg: sequential
  - sampler@test_sampler_cfg: sequential

...
```

We can replace a sampler through the commandline like so:

```shell
python train.py datamodule/sampler@datamodule.train_sampler=sequential
```






