# Description of training parameters

Here, we describe the hyperparameters used for setting up training. Please see [here](./training.md#example-config) for an example training config.

> Note: for using defaults, simply leave the field blank or don't include the key. Using `null` will initialize the value to `None` which we use to represent turning off certain features such as logging, early stopping etc. e.g
> ```YAML
> model:
>   d_model: #defaults to 1024 
>   nhead: 8
>   ...
> ```
> vs.
> ```YAML
> model:
>   d_model: null # sets model.d_model=None 
>   nhead: 8
>   ...
> ```

## `model`

This section contains all the parameters for initializing a [`GTRRunner`](../reference/dreem/models/gtr_runner.md) object

* `ckpt_path`: (`str`) the path to model `.ckpt` file. Used for resuming training.
* `d_model`: (`int`) the size of the embedding dimensions used for input into the [transformer](../reference/dreem/models/transformer.md)
* `nhead`: (`int`) the number of attention heads used in the transformer's [encoder](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerEncoderLayer)/[decoder](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerDecoderLayer) layers.
* `num_encoder_layers`: (`int`) the number of layers in the [transformer encoder block](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerEncoder)
* `num_decoder_layers`: (`int`) the number of layers in the [transformer decoder block](../reference/dreem/models/transformer.md#dreem.models.transformer.TransformerDecoder)
* `dropout`: a `float` the dropout probability used in each transformer layer
* `activation`: One of {`"relu"`, `"gelu"` `"glu"`}. Which activation function to use in the transformer.
* `return_intermediate_dec`: (`bool`) whether or not to return the output from the intermediate decoder layers.
* `norm`: (`bool`) whether or not to normalize output of encoder and decoder.
* `num_layers_attn_head`: An `int` The number of layers in the [`AttentionHead`](../reference/dreem/models/attention_head.md) block.
* `dropout_attn_head`: (`float`)  the dropout probability for the [`AttentionHead`](../reference/dreem/models/attention_head.md) block.
* `return_embedding`: (`bool`) whether to return [the spatiotemporal embeddings](../reference/dreem/models/embedding.md)
* `decoder_self_attn`: (`bool`) whether to use self attention in the decoder.
### `embedding_meta`: 

This section contains parameters for initializing the [`Embedding`](../reference/dreem/models/embedding.md) Layer.

#### `pos`

This subsection contains the parameters for initializing a Spatial [`Embedding`](../reference/dreem/models/embedding.md).

* `mode`: (`str`) One of {`"fixed"`, `"learned"`, `"None"`}. Indicates whether to use a fixed sinusoidal, learned, or no embedding.
* `n_points`: (`int`) the number of points that will be embedded.
##### Fixed Sinusoidal Params
* `temperature`: (`float`) the temperature constant to be used when computing the sinusoidal position embedding
* `normalize`: (`bool`) whether or not to normalize the positions (Only used in fixed embeddings).
* `scale`: (`float`) factor by which to scale the positions after normalizing (Only used in fixed embeddings).
##### Learned Params:
* `emb_num`: (`int`) the number of embeddings in the `self.lookup` table (Only used in learned embeddings).
* `over_boxes`: (`bool`) Whether to compute the position embedding for each bbox coordinate (`y1x1y2x2`) or the centroid + bbox size (`yxwh`).
##### `mlp_cfg`

This subsection contains [`MLP`](../reference/dreem/models/mlp.md) hyperparameters for projecting embedding to correct space. Required when `n_points > 1`, optional otherwise.

* `hidden_dims`: (`int`) The dimensionality of the MLP hidden layers.
* `num_layers`: (`int`) Number of hidden layers.
* `dropout`: (`float`) The dropout probability for each hidden layer.

Example: 
```YAML
model:
    ...
    embedding_meta:
        pos:
            ...
            n_points: 3 #could also be 1
            ...
            mlp_cfg: #cannot be null
                hidden_dims: 256
                num_layers: 3
                dropout: 0.3
```

##### Examples:
###### With MLP:
```YAML
...
model:
    ...
    embedding_meta:
        pos:
            mode: "fixed"
            normalize: true
            temperature: 10000
            scale: null
            n_points: 3 #could also be 1
            mlp_cfg: 
                hidden_dims: 256
                num_layers: 3
                dropout: 0.3
            ...
        ...
    ...
...
```
###### With no MLP
```YAML
model:
    ...
    embedding_meta:
        pos:
            mode: "fixed"
            normalize: true
            temperature: 10000
            scale: null
            n_points: 1 #must be 1
            mlp_cfg: null
        ...
    ...
...
```
#### `temp`

This subsection contains the parameters for initializing a Temporal [`Embedding`](../reference/dreem/models/embedding.md)

* `mode`: (`str`) One of {`"fixed"`, `"learned"`, `"None"`}. Indicates whether to use a fixed sinusoidal, learned, or no embedding.
##### Fixed Sinusoidal Params
* `temperature`: (`float`) the temperature constant to be used when computing the sinusoidal position embedding
##### Learned Params:
* `emb_num`: (`int`) the number of embeddings in the lookup table.
Note: See [`dreem.models.Embedding`](../reference/dreem/models/embedding.md) for additional `kwargs` that can be passed
##### Examples:
###### Fixed:
```YAML
model:
    ...
    embedding_meta:
        temp:
            mode: "fixed"
            temperature: 10000
        ...
    ...
...
```    
###### Turned Off:
```YAML
model:
    ...
    embedding_meta:
        temp: null
        ...
    ...
...
``` 
or
```YAML
model:
    ...
    embedding_meta:
        temp: 
            mode: "off" #null also accepted
            ...
        ...
    ...
...
```   
#### `embedding_meta` Example:

Putting it all together, your `embedding_meta` section should look something like this

```YAML
...
model:
    ...
    embedding_meta:
        pos:
            mode: "fixed"
            normalize: true
            temperature: 10000
            scale: null
            n_points: 3 #could also be 1
            mlp_cfg: 
                hidden_dims: 256
                num_layers: 3
                dropout: 0.3
        temp:
            mode: "fixed"
            temperature: 10000
    ...
...

```
            
### `encoder_cfg`

This section contains all the parameters for initializing a [`VisualEncoder`](../reference/dreem/models/visual_encoder.md) model.

* `model_name`: (`str`) Thhe name of the visual encoder backbone to be used. When using `timm` as a backend, all models in `timm.list_model` are available. However, when using `torchvision` as a backend, only `resnet`s are available for now.
* `backend`: (`str`) Either `"timm"` or `"torchvision"`. Indicates which deep learning library to use for initializing the visual encoder
* `in_chans`: (`int`)  the number of input channels input images contain. Mostly used for multi-anchor crops
* `pretrained`: (`bool`) Whether or not to use a pretrained backbone or initialize from random

> Note: For more advanced users, see [`timm.create_model`](https://timm.fast.ai/create_model) or [`torchvision.models.resnet`](https://pytorch.org/vision/stable/models/resnet.html) for additional `kwargs` that can be passed to the visual encoder.

#### Example:
##### `timm`:
```YAML
...
model:
    ...
    encoder_cfg:
        model_name: "resnet18"
        backend: "timm"
        in_chans: 3
        pretrained: false
        ...
    ...
...
```
##### `torchvision`:
```YAML
...
model:
    ...
    encoder_cfg:
        model_name: "resnet32"
        backend: "torchvision"
        in_chans: 3
        pretrained: false
        ...
    ...
...
```
### `model` Example:
Putting it all together your `model` config section will look something like this
```YAML
...
model:
  ckpt_path: null
  encoder_cfg: 
    model_name: "resnet18"
    backend: "timm"
    in_chans: 3
  d_model: 1024
  nhead: 8
  num_encoder_layers: 1
  num_decoder_layers: 1
  dropout: 0.1
  activation: "relu"
  return_intermediate_dec: False
  norm: False
  num_layers_attn_head: 2
  dropout_attn_head: 0.1
  embedding_meta: 
    pos:
        mode: "fixed"
        normalize: true
    temp:
        mode: "fixed"
  return_embedding: False
  decoder_self_attn: False
...
```
## `loss`

This section contains parameters for the [Association Loss function](../reference/dreem/training/losses.md#dreem.training.losses.AssoLoss)

* `neg_unmatched` a bool whether to set unmatched objects to the background
* `epsilon`: A small `float` used for numeric precision to prevent dividing by zero
* `asso_weight`: (`float`) how much to weight the association loss by

### Examples:
```YAML
...
loss:
    neg_unmatched: false
    epsilon: 1e-8
    asso_weight: 1.0
...
```
## `optimizer`

This section contains the parameters for initializing the training optimizer

* `name`: (`str`) representation of the optimizer. 
    > See [`torch.optim`](https://pytorch.org/docs/stable/optim.html#algorithms) for available optimizers.(`name` must match the optimizer name exactly (case-sensitive)).

> Below, we list the arguments we use for [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) which is the optimizer we use and is our default. For more advanced users please see the respective pytorch documentation page for the arguments expected in your requested optimizer.

* `lr`: (`float`) learning rate
* `betas`: (`tuple[float, float]`) coefficients used for computing running averages of gradient and its square
* `eps`: (`float`): term added to the denominator to improve numerical stability
* `weight_decay`: (`float`) weight decay ($L_2$ penalty)

### Examples:
Here we provide a couple examples for different optimizers:
#### [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
```YAML
...
optimizer:
    name: "Adam"
    lr: 0.001
    betas: [0.9, 0.999]
    eps: 1e-8
    weight_decay: 0.01
    ...
...
```
#### [`Stochastic Gradient Descent`](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)
```YAML
...
optimizer:
    name: "SGD" #must match `torch.optim` class name
    lr: 0.001
    momentum: 0.9
    weight_decay: 0.01
    dampening: 1e-8
    nesterov: true
    ...
...
```
## `scheduler`

This section contains parameters for initializing the learning rate scheduler.

* `name`: (`str`) Representation of the scheduler. 
    > See [`torch.optim.lr_scheduler`](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate) for available schedulers. `name` must match the scheduler name exactly (case-sensitive).

> Below, we list the arguments we use for [`ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau) which is the scheduler we use and is our default. For more advanced users please see the respective pytorch documentation page for the arguments expected in your requested scheduler.

* `mode`: (`str`) One of {`"min"`, `"max"`}. In `min` mode, lr will be reduced when the quantity monitored has stopped decreasing; in `max` mode it will be reduced when the quantity monitored has stopped increasing.
* `factor`: (`float`) Factor by which the learning rate will be reduced. `new_lr = lr * factor`
* `patience`: (`int`) The number of allowed epochs with no improvement after which the learning rate will be reduced.
* `threshold`: (`float`) Threshold for measuring the new optimum, to only focus on significant changes. 
* `threshold_mode`: (`str`)  One of {`"rel"`, "`abs`"}. In `rel` mode, `dynamic_threshold = best * ( 1 + threshold )` in `max` mode or `best * ( 1 - threshold )` in `min` mode. In `abs` mode, `dynamic_threshold = best + threshold` in `max` mode or `best - threshold` in `min` mode.

### Examples:
Here we give a couple examples of configs for different schedulers:

#### [`Reduce Learning Rate on Plateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
```YAML
...
scheduler:
  name: "ReduceLROnPlateau" #must match torch.optim class name
  mode: "min"
  factor: 0.5
  patience: 10
  threshold: 1e-4
  threshold_mode: "rel"
  ...
...
```
#### [`Cosine Annealing with Warm Restarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
```YAML
...
scheduler:
    name: "CosineAnnealingWarmRestarts"
    T_0: 10
    T_mult: 1
    eta_min: 0
    last_epoch: 50
    verbose: True
...
```
## `tracker`:

This section contains parameters for initializing the [`Tracker`](../reference/dreem/inference/tracker.md)

* `window_size`: the size of the window used during sliding inference.
* `use_vis_feats`: Whether or not to use visual feature extractor.
* `overlap_thresh`: the trajectory overlap threshold to be used for assignment.
* `mult_thresh`: Whether or not to use weight threshold.
* `decay_time`: weight for `decay_time` postprocessing.
* `iou`: Either `{None, '', "mult" or "max"}`. Whether to use multiplicative or max iou reweighting.
* `max_center_dist`: distance threshold for filtering trajectory score matrix.
* `persistent_tracking`: whether to keep a buffer across chunks or not.
* `max_gap`: the max number of frames a trajectory can be missing before termination.
* `max_tracks`: the maximum number of tracks that can be created while tracking.
    We force the tracker to assign instances to a track instead of creating a new track if `max_tracks` has been reached.

### Examples:
```YAML
...
tracker:
    window_size: 8
    overlap_thresh: 0.01
    mult_thresh: false
    decay_time: 0.9
    iou: "mult"
    max_center_dist: 0.1
    ...
...
```
## `runner`

This section contains parameters for how to handle training/validation/testing

### `metrics`

This section contains config for which metrics to compute during training/validation/testing. See [`pymotmetrics.list_metrics`](https://github.com/cheind/py-motmetrics) for available metrics.

Should have a `train`, `val` and `test` key with corresponding list of metrics to compute during training.

#### Examples:
##### Only computing the loss:
```YAML
...
runner:
    ...
    metrics:
        train: []
        val: []
        test: []
    ...
...
```
##### Computing `num_switches` during validation:
```YAML
...
runner:
    ...
    metrics:
        train: []
        val: ["num_switches"]
        test: []
    ...
...
```
##### Computing `num_switches` and  `mota` during testing:
```YAML
...
runner:
    ...
    metrics:
        train: []
        val: ["num_switches"]
        test: ["num_switches", "mota"]
    ...
...
```
### `persistent_tracking`

This section indicates whether or not to track across chunks during training/validation/testing

Should have a `train`, `val` and `test` key with a corresponding `bool` whether to use persistent tracking.
`persistent_tracking` should almost always be `False` during training. During validation and testing it may depend on whether you are testing on full videos or subsampled clips

#### Examples:
```YAML
...
runner
    ...
    persistent_tracking:
        train: false
        val: false # assuming we validate on a subsample of clips
        test: true # assuming we test on a contiguous video.
```

## `dataset`

This section contains the params for initializing the datasets for training. Requires a `train_dataset` and optionally `val_dataset`, `test_dataset` keys. 

### [`BaseDataset`](../reference/dreem/datasets/base_dataset.md) args

* `padding`: An `int` representing the amount of padding to be added to each side of the bounding box size
* `crop_size`: (`int`|`tuple`) the size of the bounding box around which a crop will form.
* `chunk`: Whether or not to chunk videos into smaller clips to feed to model
* `clip_length`: the number of frames in each chunk
* `mode`: `train` or `val`. Determines whether this dataset is used for training or validation.
* `n_chunks`: Number of chunks to subsample from. Can either a fraction of the dataset (ie `(0,1.0]`) or number of chunks
* `seed`: set a seed for reproducibility
* `gt_list`: An optional path to .txt file containing ground truth for cell tracking challenge datasets.

#### `dir`:
This section allows you to pass a directory rather than paths to labels/videos individually

* `path`: The path to the dir where the data is stored (recommend absolute path)
* `labels_suffix`: (`str`) containing the file extension to search for labels files. e.g. `.slp`, `.csv`, or `.xml`.
* `vid_suffix`: (`str`) containing the file extension to search for video files e.g `.mp4`, `.avi` or `.tif`.
##### Examples:
```YAML
...
dataset:
    ...
    {MODE}_dataset:
        dir:
            path: "/path/to/data/dir/mode"
            labels_suffix: ".slp"
            vid_suffix: ".mp4"
        ...
    ...
...
```
#### `augmentations`:

This subsection contains params for albumentations. See [`albumentations`](https://albumentations.ai) for available visual augmentations. Other available augmentations include `NodeDropout` and `InstanceDropout`. Keys must match augmentation class name exactly and contain subsections with parameters for the augmentation

##### Example
```YAML
augmentations: 
    Rotate:
        limit: 45
        p: 0.3
    ...
    MotionBlur:
        blur_limit: [3,7]
        p: 0.3
```
### [`SleapDataset`](../reference/dreem/datasets/sleap_dataset.md) Args:
* `slp_files`: (`str`) a list of .slp files storing tracking annotations
* `video_files`: (`str`) a list of paths to video files
* `anchors`: (`str` | `list` | `int`) One of:
    * a string indicating a single node to center crops around
    * a list of skeleton node names to be used as the center of crops
    * an int indicating the number of anchors to randomly select
    If unavailable then crop around the midpoint between all visible anchors.
* `handle_missing`: how to handle missing single nodes. one of [`"drop"`, `"ignore"`, `"centroid"`].
    * if `drop` then we dont include instances which are missing the `anchor`.
    * if `ignore` then we use a mask instead of a crop and nan centroids/bboxes.
    * if `centroid` then we default to the pose centroid as the node to crop around.
### [`MicroscopyDataset`](../reference/dreem/datasets/microscopy_dataset.md)
* `videos`: (`list[str | list[str]]`) paths to raw microscopy videos
* `tracks`: (`list[str]`) paths to trackmate gt labels (either `.xml` or `.csv`)
* `source`: file format of gt labels based on label generator. Either `"trackmate"` or `"isbi"`.
### [`CellTrackingDataset`](../reference/dreem/datasets/cell_tracking_dataset.md)
* `raw_images`: (`list[list[str] | list[list[str]]]`) paths to raw microscopy images
* `gt_images`: (`list[list[str] | list[list[str]]]`) paths to gt label images
* `gt_list`: (`list[str]`) An optional path to .txt file containing gt ids stored in cell
                tracking challenge format: `"track_id", "start_frame",
                "end_frame", "parent_id"`
### `dataset` Examples
#### [`SleapDataset`](../reference/dreem/datasets/sleap_dataset.md)
```YAML
...
dataset:
    train_dataset:
        slp_files: ["/path/to/train/labels1.slp", "/path/to/train/labels2.slp", ..., "/path/to/train/labelsN.slp"]
        video_files: ["/path/to/train/video1.mp4", "/path/to/train/video2.mp4", ..., "/path/to/train/videoN.mp4"]
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        anchors: ["node1", "node2", ..."node_n"]
        handle_missing: "drop"
        augmentations: 
            Rotate:
                limit: 45
                p: 0.3
            ...
            MotionBlur:
                blur_limit: [3,7]
                p: 0.3
        ...
    val_dataset:
        slp_files: ["/path/to/val/labels1.slp", "/path/to/val/labels2.slp", ..., "/path/to/val/labelsN.slp"]
        video_files: ["/path/to/val/video1.mp4", "/path/to/val/video2.mp4", ..., "/path/to/val/videoN.mp4"]
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        anchors: ["node1", "node2", ..."node_n"]
        handle_missing: "drop"
        ... # we don't include augmentations bc usually you shouldnt use augmentations during val/test
    test_dataset:
        slp_files: ["/path/to/test/labels1.slp", "/path/to/test/labels2.slp", ..., "/path/to/test/labelsN.slp"]
        video_files: ["/path/to/test/video1.mp4", "/path/to/test/video2.mp4", ..., "/path/to/test/videoN.mp4"]
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        anchors: ["node1", "node2", ..."node_n"]
        handle_missing: "drop"
        ... # we don't include augmentations bc usually you shouldnt use augmentations during val/test
...
```
#### [`MicroscopyDataset`](../reference/dreem/datasets/microscopy_dataset.md)
```YAML
dataset:
    train_dataset:
        tracks: ["/path/to/train/labels1.csv", "/path/to/train/labels2.csv", ..., "/path/to/train/labelsN.csv"]
        videos: ["/path/to/train/video1.tiff", "/path/to/train/video2.tiff", ..., "/path/to/train/videoN.tiff"]
        source: "trackmate"
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        augmentations: 
            Rotate:
                limit: 45
                p: 0.3
            ...
            MotionBlur:
                blur_limit: [3,7]
                p: 0.3
        ...
    val_dataset:
        tracks: ["/path/to/val/labels1.csv", "/path/to/val/labels2.csv", ..., "/path/to/val/labelsN.csv"]
        video: ["/path/to/val/video1.tiff", "/path/to/val/video2.tiff", ..., "/path/to/val/videoN.tiff"]
        source: "trackmate"
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        ... # we don't include augmentations bc usually you shouldnt use augmentations during val/test
    test_dataset:
        tracks: ["/path/to/test/labels1.csv", "/path/to/test/labels2.csv", ..., "/path/to/test/labelsN.csv"]
        videos: ["/path/to/test/video1.tiff", "/path/to/test/video2.tiff", ..., "/path/to/test/videoN.tiff"]
        source: "trackmate"
        padding: 5
        crop_size: 128 
        chunk: True
        clip_length: 32
        ... # we don't include augmentations bc usually you shouldnt use augmentations during val/test
```
## `dataloader`

This section outlines the params needed for the dataloader. Should have a `train_dataloader` and optionally `val_dataloader`/`test_dataloader` keys. 
> Below we list the args we found useful/necessary for the dataloaders. For more advanced users see [`torch.utils.data.Dataloader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for more ways to initialize the dataloaders

* `shuffle`: (`bool`) Set to `True` to have the data reshuffled at every epoch (during training, this should always be `True` and during val/test usually `False`) 
* `num_workers`: (`int`) How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.

### Example
```YAML
...
dataloader:
    train_dataloader:
        shuffle: true
        num_workers: 4
    val_dataloader: # we leave out the `shuffle` field as default=`False` which is what we want
        num_workers: 4
    test_dataloader: # we leave out the `shuffle` field as default=`False` which is what we want
        num_workers: 4
```

## `logging`:
This section sets up logging for the training job. 

* `logger_type`: (`str`) Which logger to use. Available loggers are {`"CSVLogger"`, `"TensorBoardLogger"`,`"WandbLogger"`}

> Below we list the arguments we found useful for the [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) as this is the logger we use and recommend. Please see the documentation for the corresponding logger at [`lightning.loggers`](https://lightning.ai/docs/pytorch/stable/api_references.html#loggers) for respective available parameters.

* `name`: (`str`) A short display name for this run, which is how you'll identify this run in the UI.
* `save_dir`: (`str`) An absolute path to a directory where metadata will be stored. 
* `version`: (`str`) A unique ID for this run, used for resuming. It must be unique in the project, and if you delete a run you can't reuse the ID.
* `project`: (`str`)  The name of the project where you're sending the new run.
* `log_model`: (`str`) Log checkpoints created by `ModelCheckpoint` as W&B artifacts
* `group`: (`str`) Specify a group to organize individual runs into a larger experiment
* `entity`: (`str`) An entity is a username or team name where you're sending runs
* `notes`: (`str`) A longer description of the run, like a `-m `commit message in git.

> See [`wandb.init()`](https://docs.wandb.ai/ref/python/init) and [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb) for more fine-grained config args.

### Examples:
Here we provide a couple examples for different available loggers
#### [`wandb`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html#module-lightning.pytorch.loggers.wandb)
```YAML
...
logging:
  logger_type: "WandbLogger"
  name: "example_train"
  entity: "example_user"
  job_type: "train"
  notes: "Example train job"
  dir: "./logs"
  group: "example"
  save_dir: './logs'
  project: "GTR"
  log_model: "all"
  ...
...
```

#### [`csv logger`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.csv_logs.html#module-lightning.pytorch.loggers.csv_logs):
```YAML
...
logging:
    save_dir: "./logs"
    name: "example_train.csv"
    version: 1
    flush_logs_every_n_steps: 1
    ...
...
``` 
## `early_stopping`

This section configures early stopping for training runs. 

> Below we provide descriptions of the arguments we found useful for EarlyStopping. For advanced users, see [`lightning.callbacks.EarlyStopping](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.pytorch.callbacks.EarlyStopping) for available arguments for more fine grained control

* `monitor` (`str`): quantity to be monitored.
* `min_delta` (`float`): minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than or equal to min_delta, will count as no improvement.
* `patience` (`int`): number of checks with no improvement after which training will be stopped. 
* `mode` (`str`): one of 'min', 'max'. In 'min' mode, training will stop when the quantity monitored has stopped decreasing and in 'max' mode it will stop when the quantity monitored has stopped increasing.
* `check_finite` (`bool`): When set True, stops training when the monitor becomes NaN or infinite.
* `stopping_threshold` (`float`): Stop training immediately once the monitored quantity reaches this threshold.
* `divergence_threshold` (`float`): Stop training as soon as the monitored quantity becomes worse than this threshold.

### Example:
```YAML
...
early_stopping:
  monitor: "val_loss"
  min_delta: 0.1
  patience: 10
  mode: "min"
  check_finite: true
  stopping_threshold: 1e-8
  divergence_threshold: 30
  ...
...
```

## `checkpointing`

This section enables model checkpointing during training

* `monitor`: A list of metrics to save best models for. Usually should be `"val_{METRIC}"` notation.
    > Note: We initialize a separate [`ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint) for each metric to monitor.
    > This means that you'll save at least $|monitor|$ checkpoints at the end of training.

> Below we describe the arguments we found useful for checkpointing. For more fine grained control see [`lightning.callbacks.ModelCheckpoint`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint) for available checkpointing params and generally more info on how `lightning` sets up checkpoints

* `dirpath`: (`str`) Directory to save the models. If left empty then we first try to save to `./models/[GROUP]/[NAME]` or `./models/[NAME]` if logger is `wandb` otherwise we just save to `./models` 
* `save_last`: (`bool`): When `True`, saves a last.ckpt copy whenever a checkpoint file gets saved. Can be set to 'link' on a local filesystem to create a symbolic link. This allows accessing the latest checkpoint in a deterministic manner.
* `save_top_k`: (`int`): if `save_top_k == k`, the best k models according to the quantity monitored will be saved. if `save_top_k == 0`, no models are saved. if `save_top_k == -1`, all models are saved. (Recommend -1)
* `every_n_epochs`: (`int`) Number of epochs between checkpoints. This value must be `None` or non-negative. To disable saving top-k checkpoints, set `every_n_epochs = 0`. This argument does not impact the saving of `save_last=True` checkpoints.

### Example:
```YAML
...
checkpointing:
    monitor: ["val_loss", "val_num_switches"] #saves a model for best validation loss and a model for best validation switch count separately
    dirpath: "./models/example_run"
    save_last: true # will always save the best run
    save_top_k: -1
    every_n_epochs: 10 # saves the every 10th model regardless of if its the best.
    ...
...
```

## `trainer`

This section configures the `lightning.Trainer` object for training. 
> Below we describe the arguments we found useful for the `Trainer`. If you're an advanced user, Please see `lightning.Trainer`(https://lightning.ai/docs/pytorch/stable/common/trainer.html) for more fine grained control and how the `trainer` works in general

* `accelerator`: (`str`) Supports passing different accelerator types `(“cpu”, “gpu”, “tpu”, “ipu”, “hpu”, “mps”, “auto”)` as well as custom accelerator instances.
* `strategy`: (`str`) Supports different training strategies with aliases as well custom strategies
* `devices`: (`list[int]` | `str`| `int`)`The devices to use. Can be set to:
    * a positive number (`int` | `str`) 
    * a sequence of device indices (`list` | `str`), 
    * the value `-1` to indicate all available devices should be used
    *  "auto" for automatic selection based on the chosen accelerator 
* `fast_dev_run`: (`int` | `bool`) Runs `n` (if set to `n` (`int`)) else `1` (if set to `True`) batch(es) of train, val and test to find any bugs (ie: a sort of unit test).
* `check_val_every_n_epoch`: (`int`) Perform a validation loop every after every `N` training epochs
* `enable_checkpointing`: (`bool`) If `True`, enable checkpointing. It will configure a default `ModelCheckpoint` callback if there is no user-defined `ModelCheckpoint` in callbacks.
* `gradient_clip_val`:  (`float`) The value at which to clip gradients
* `limit_train_batches`: (`int` | `float`) How much of training dataset to check (`float` = fraction, `int` = num_batches) (mostly for debugging)
* `limit_test_batches`: (`int` | `float`) How much of test dataset to check (`float` = fraction, `int` = num_batches). (mostly for debugging)
* `limit_val_batches`: (`int` | `float`) How much of validation dataset to check (`float` = fraction, `int` = num_batches) (mostly for debugging)
* `limit_predict_batches`: (`int` | `float`) How much of prediction dataset to check (`float` = fraction, `int` = num_batches)
* `log_every_n_steps`:  (`int`) How often to log within steps
* `max_epochs`: (`int`) Stop training once this number of epochs is reached. To enable infinite training, set `max_epochs` = -1.
* `min_epochs`: (`int`) Force training for at least these many epochs

### Examples:
```YAML
trainer:
  check_val_every_n_epoch: 1
  enable_checkpointing: true
  gradient_clip_val: null
  limit_train_batches: 1.0
  limit_test_batches: 1.0
  limit_val_batches: 1.0
  log_every_n_steps: 1
  max_epochs: 100
  min_epochs: 10
```

## `view_batch`

This section allows you to visualize the data before training

* `enable`: (`bool`) whether or not to view a batch
* `num_frames`: (`int`) The number of frames in the batch to visualize
* `no_train`: (`bool`)  whether or not to train after visualization is complete

### Examples:
#### Off
```YAML
view_batch:
  enable: False
  num_frames: 0 #this arg can be anything
  no_train: False #This can be false
```
#### On, no training:
```YAML
view_batch:
  enable: False
  num_frames: 32 #this arg can be anything
  no_train: True #training will not occur
```
#### On, with training: 
```YAML
view_batch:
  enable: False
  num_frames: 32 #this arg can be anything
  no_train: True #training will not occur
```

# Example Config
## Base Config
```YAML
--8<-- "dreem/training/configs/base.yaml"
```
## Override Config
```YAML
--8<-- "dreem/training/configs/params.yaml"