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


class FileVideoStream:
	def __init__(self, path, transform=None, queue_size=16):
		# initialize the file video stream along with the boolean
		# used to indicate if the thread should be stopped or not
		self.stream = cv2.VideoCapture(path)
		self.stopped = False
		self.transform = transform

		# initialize the queue used to store frames read from
		# the video file
		self.Q = Queue(maxsize=queue_size)
		# intialize thread
		self.thread = Thread(target=self.update, args=())
		self.thread.daemon = True

	def start(self):
		# start a thread to read frames from the file video stream
		self.thread.start()
		return self

	def update(self):
		# keep looping infinitely
		while True:
			# if the thread indicator variable is set, stop the
			# thread
			if self.stopped:
				break

			# otherwise, ensure the queue has room in it
			if not self.Q.full():
				# read the next frame from the file
				(grabbed, frame) = self.stream.read()

				# if the `grabbed` boolean is `False`, then we have
				# reached the end of the video file
				if not grabbed:
					self.stopped = True
					
				# if there are transforms to be done, might as well
				# do them on producer thread before handing back to
				# consumer thread. ie. Usually the producer is so far
				# ahead of consumer that we have time to spare.
				#
				# Python is not parallel but the transform operations
				# are usually OpenCV native so release the GIL.
				#
				# Really just trying to avoid spinning up additional
				# native threads and overheads of additional
				# producer/consumer queues since this one was generally
				# idle grabbing frames.
				if self.transform:
					frame = self.transform(frame)

				# add the frame to the queue
				self.Q.put(frame)
			else:
				time.sleep(0.1)  # Rest for 10ms, we have a full queue

		self.stream.release()

	def read(self):
		# return next frame in the queue
		return self.Q.get()

	# Insufficient to have consumer use while(more()) which does
	# not take into account if the producer has reached end of
	# file stream.
	def running(self):
		return self.more() or not self.stopped

	def more(self):
		# return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
		tries = 0
		while self.Q.qsize() == 0 and not self.stopped and tries < 5:
			time.sleep(0.1)
			tries += 1

		return self.Q.qsize() > 0

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True
		# wait until stream resources are released (producer thread might be still grabbing frame)
		self.thread.join()


class Q:
    def __init__(self, buffer_size=16, size=(160, 120)):
        self.buffer_size=buffer_size
        self.size=size
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3, size[0])
        self.capture.set(4, size[1])
        self.queue = []

    def read(self):
        _, frame = self.capture.read()
        h, w, c = frame.shape
        frame = np.expand_dims(frame, 0)
        frame = frame.reshape((c, 1, h, w))
        self.queue.append(frame)
        if len(self.queue) > self.buffer_size:
            self.queue.pop(0)
        
        frame_stack = np.repeat(np.stack(self.queue), 2, axis=1)
        frame_stack = torch.FloatTensor(frame_stack).view(2, 3, -1, self.size[0], self.size[1])
        # frame_stack = frame_stack.unsqueeze(0)
        return frame_stack
    



def main(config, resume):
    # setup data_loader instances
    # data_loader = getattr(module_data, config['train_data_loader']['type'])(
    #     config['train_data_loader']['args']['data_dir'],
    #     batch_size=32,
    #     shuffle=False,
    #     val_split=0.0,
    #     split='test',
    #     num_workers=2
    # )
    print("Opening video capture...")
    # cap = cv2.VideoCapture(0)
    
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
    q = Q(16)
    fps = FPS().start()
    transform = module_data.eval_transform

    with torch.no_grad():
        while True:
            frames = q.read()
            frames = transform(frames)
            print('READING FRAMES', frames.size())
            data = frames.to(device)
    #         print(data.size())
            output = model(data)
            print(output)
            #
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

    parser.add_argument('-r', '--resume', default=None, type=str,
                           help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='0,1', type=str,
                           help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    import pprint as pp

    if args.resume:
        config = torch.load(args.resume)['config']
        pp.pprint(config)
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
        config['n_gpu'] = len(args.device.split(','))
    main(config, args.resume)
