# Gesture Recognition Models

<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

* [Gesture Recognition Models](#gesture-recognition-models)
	* [Requirements](#requirements)
	* [Features](#features)
	* [Folder Structure](#folder-structure)
	* [Usage](#usage)
		* [Using config files](#using-config-files)
		* [Resuming from checkpoints](#resuming-from-checkpoints)
    * [Using Multiple GPU](#using-multiple-gpu)
		* [Loss and metrics](#loss-and-metrics)
			* [Multiple metrics](#multiple-metrics)
		* [Validation data](#validation-data)
		* [Checkpoints](#checkpoints)
    * [TensorboardX Visualization](#tensorboardx-visualization)
	* [TODOs](#todos)
	* [License](#license)
	* [Acknowledgments](#acknowledgments)

<!-- /code_chunk_output -->

## Requirements
* Python >= 3.5
* PyTorch >= 1.0
* tqdm 
* tensorboard >= 1.7.0 
* tensorboardX >= 1.2 

## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary.

## Folder Structure
  ```
  gesture_recognition_models/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── demo.py - webcam demo of trained model
  ├── config.json - config file
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py - abstract base class for data loaders
  │   ├── base_model.py - abstract base class for models
  │   └── base_trainer.py - abstract base class for trainers
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── data_loaders.py - data loader class
  │   └── transforms.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── loss.py
  │   ├── metric.py
  │   └── model.py
  │
  ├── saved/ - default checkpoints folder
  │   └── runs/ - default logdir for tensorboardX
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  └── utils/
      ├── util.py
      ├── logger.py - class for train logging
      ├── lrfinder.py - class for finding good initial lr
      └── visualization.py - class for tensorboardX visualization support
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python3 train.py -c config.json` to run code.
Try `python3 test.py --resume ...` to test a model
Try `python3 demo.py --resume ...` to run the demo


Add addional configurations if you need.

### Using config files
Modify the configurations in `.json` config files, then run:

  ```
  python3 train.py --config config.json
  ```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:

  ```
  python3 train.py --resume path/to/checkpoint
  ```

### Using Multiple GPU
You can enable multi-GPU training by setting `n_gpu` argument of the config file to larger number.
If configured to use smaller number of gpu than available, first n devices will be used by default.
Specify indices of available GPUs by cuda environmental variable.
  ```
  python3 train.py --device 2,3 -c config.json
  ```
  This is equivalent to
  ```
  CUDA_VISIBLE_DEVICES=2,3 python3 train.py -c config.py
  ```

#### Metrics
Metric functions are located in 'model/metric.py'.

You can monitor multiple metrics by providing a list in the configuration file, e.g.:
  ```json
  "metrics": ["my_metric", "my_metric2"],
  ```

### Additional logging
If you have additional information to be logged, in `_train_epoch()` of your trainer class, merge them with `log` as shown below before returning:

  ```python
  additional_log = {"gradient_norm": g, "sensitivity": s}
  log = {**log, **additional_log}
  return log
  ```
  
### Testing
You can test trained model by running `test.py` passing path to the trained checkpoint by `--resume` argument.

### Validation data
To split validation data from a data loader, call `BaseDataLoader.split_validation()`, it will return a validation data loader, with the number of samples according to the specified ratio in your config file.

**Note**: the `split_validation()` method will modify the original data loader
**Note**: `split_validation()` will return `None` if `"validation_split"` is set to `0`

### Checkpoints
You can specify the name of the training session in config files:
  ```json
  "name": "MNIST_LeNet",
  ```

The checkpoints will be saved in `save_dir/name/timestamp/checkpoint_epoch_n`, with timestamp in mmdd_HHMMSS format.

A copy of config file will be saved in the same folder.

**Note**: checkpoints contain:
  ```python
  {
    'arch': arch,
    'epoch': epoch,
    'logger': self.train_logger,
    'state_dict': self.model.state_dict(),
    'optimizer': self.optimizer.state_dict(),
    'monitor_best': self.monitor_best,
    'config': self.config
  }
  ```

### TensorboardX Visualization
This template supports [TensorboardX](https://github.com/lanpa/tensorboardX) visualization.
* **TensorboardX Usage**

1. **Install**

    Follow installation guide in [TensorboardX](https://github.com/lanpa/tensorboardX).

2. **Run training** 

    Set `tensorboardX` option in config file true.

3. **Open tensorboard server** 

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at `http://localhost:6006`

By default, values of loss and metrics specified in config file, and input image will be logged.
If you need more visualizations, use `add_scalar('tag', data)`, `add_image('tag', image)`, etc in the `trainer._train_epoch` method.
`add_something()` methods in this template are basically wrappers for those of `tensorboardX.SummaryWriter` module. 

**Note**: You don't have to specify current steps, since `WriterTensorboardX` class defined at `logger/visualization.py` will track current steps.

## Contributing
Feel free to contribute any kind of function or enhancement, here the coding style follows PEP8

Code should pass the [Flake8](http://flake8.pycqa.org/en/latest/) check before committing.

## TODOs
- [ ] Iteration-based training (instead of epoch-based)
- [ ] Multiple optimizers
- [ ] Configurable logging layout, checkpoint naming
- [ ] `visdom` logger support
- [x] `tensorboardX` logger support
- [x] Adding command line option for fine-tuning
- [x] Multi-GPU support
- [x] Update the example to PyTorch 0.4
- [x] Learning rate scheduler
- [x] Deprecate `BaseDataLoader`, use `torch.utils.data` instesad
- [x] Load settings from `config` files

## License
This project is licensed under the MIT License. See  LICENSE for more details

## Acknowledgments
This project is inspired by the project [Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template) by [Mahmoud Gemy](https://github.com/MrGemy95)
