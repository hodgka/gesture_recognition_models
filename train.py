import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
# from data_loader.data_loaders import VideoFolder, transforms, default_loader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from trainer import Trainer
from utils import Logger#, LRFinder


def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume, args):
    train_logger = Logger()

    # setup data_loader instances
    print(dir(module_data))
    train_data_loader = get_instance(module_data, 'train_data_loader', config)
    val_data_loader = get_instance(module_data, 'val_data_loader', config)
    # train_dataset = VideoFolder(root="/u/big/trainingdata/20BNJESTER/20bn-jester-v1/",
    #                      csv_file_input="/u/big/trainingdata/20BNJESTER/jester-v1-train.csv",
    #                      csv_file_labels="/u/big/trainingdata/20BNJESTER/jester-v1-labels.csv",
    #                      clip_size=18,
    #                      nclips=1,
    #                      step_size=2,
    #                      is_val=False,
    #                      transform=transform,
    #                      loader=default_loader)
    # val_dataset = VideoFolder(root="/u/big/trainingdata/20BNJESTER/20bn-jester-v1/",
    #                      csv_file_input="/u/big/trainingdata/20BNJESTER/jester-v1-validation.csv",
    #                      csv_file_labels="/u/big/trainingdata/20BNJESTER/jester-v1-labels.csv",
    #                      clip_size=18,
    #                      nclips=1,
    #                      step_size=2,
    #                      is_val=True,
    #                      transform=transform,
    #                      loader=default_loader)

    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config['train_data_loader']['args']['batch_size'], shuffle=True,
    #     num_workers=5, pin_memory=True)

    # val_data_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=config['val_data_loader']['args']['batch_size'], shuffle=False,
    #     num_workers=5, pin_memory=True)
    # train_data_loader.n_samples = len(train_dataset)
    # val_data_loader.n_samples = len(val_dataset)

    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
    lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)

    if args.find_lr:
        lr_finder = LRFinder(model, optimizer, loss, device='cuda')
        lr_finder.range_test(train_data_loader, end_lr=1, num_iter=100)
        lr_finder.plot()
        lr_finder.reset()
    trainer = Trainer(model, loss, metrics, optimizer, 
                      resume=resume,
                      config=config,
                      data_loader=train_data_loader,
                      val_data_loader=val_data_loader,
                      lr_scheduler=lr_scheduler,
                      train_logger=train_logger)

    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gesture Recognition Model')
    parser.add_argument('-c', '--config', default='config.json', type=str,
                           help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='0', type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('-f', '--find_lr', action='store_true',
                           help='Enable learning rate finder phase (default: False)')
    
    args = parser.parse_args()

    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
    
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
        config['n_gpu'] = len(args.device.split(','))
    print(config['n_gpu'])
    main(config, args.resume, args)
