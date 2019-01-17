import os
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from train import get_instance
import cv2
import datetime
import numpy as np
import csv

from queue import Queue
from threading import Thread
from collections import deque

class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class Q:
    def __init__(self, buffer_size=18, size=(160, 120)):
        self.buffer_size=buffer_size
        self.size=size
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FPS, 12)
        self.capture.set(3, size[0])
        self.capture.set(4, size[1])
        self.queue = []

    def read(self):
        _, frame = self.capture.read()
        frame = frame / 255.
        h, w, c = frame.shape
        frame = np.expand_dims(frame, 0)
        frame = np.transpose(frame, (3, 0, 1, 2))
        self.queue.append(frame)
        if len(self.queue) > self.buffer_size:
            self.queue.pop(0)
        
        frame_stack = np.stack(self.queue)
        frame_stack = torch.FloatTensor(frame_stack)
        frame_stack = frame_stack.permute(2, 1, 0, 3, 4)
        return frame_stack
    
    def show_frame(self, image):
        _, c, _, h, w = image.shape
        im = image[0, :, self.buffer_size-1, :, :].numpy().transpose((1, 2, 0))
        
        im = cv2.resize(im, (0, 0), fx=4, fy=4)
        # im = cv2.cvtColor(im,cv2.COLOR_RGB2BGR)
        cv2.imshow('stream', im)
        cv2.waitKey(1)



def main(config, resume):
    print("Opening video capture...")
    
    print("Building model...")
    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # load state dict
    print("Loading Model...")
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        print("Running on {} GPUS...".format(config['n_gpu']))
        model = torch.nn.DataParallel(model)
        
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    print("Model Loaded...")
    print()
    print()
    q = Q(18)
    fps = FPS().start()
    transform = module_data.eval_transform
    for i in range(18):
        _ = q.read()
    
    with open('/u/big/trainingdata/20BNJESTER/jester-v1-labels.csv', 'r') as f:
        pred_to_class = {}
        csv_reader = csv.reader(f)
        for i, row in enumerate(csv_reader):
            pred_to_class[i] = row[0]
       
    with torch.no_grad():
        while True:
            frames = q.read()
            q.show_frame(frames)
            frames = transform(frames)

            data = frames.to(device)
            output = model(data)
            ind = torch.argmax(output).item()
            print(pred_to_class[ind] + ' '*50, end='\r', flush=True)

            # print(torch.exp(output))

            # save sample images, or do something with output here
            #
            
            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            # # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size
            

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
    # print(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default='resnet10/GesturePredictor/0114_182708/model_best.pth', type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='0,1', type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    import pprint as pp

    # if args.resume:
    config = torch.load(args.resume)['config']
    pp.pprint(config)
    # if args.device:
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    config['n_gpu'] = len(args.device.split(','))
    main(config, args.resume)
