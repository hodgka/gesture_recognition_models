import os,sys,inspect
parentdir = '/u/big/workspace_hodgkinsona/gesture_recognition/gesture_recognition_models'
sys.path.insert(0,parentdir) 

import csv
import glob
from collections import namedtuple

from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import numpy as np
import torch
import time

from PIL import Image
import numbers 
# # from utils import save_images_for_debug

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG']

ListDataJpeg = namedtuple('ListDataJpeg', ['id', 'label', 'path'])

class JpegDataset(object):

    def __init__(self, csv_path_input, csv_path_labels, data_root):
        self.csv_data = self.read_csv_input(csv_path_input, data_root)
        self.classes = self.read_csv_labels(csv_path_labels)
        self.classes_dict = self.get_two_way_dict(self.classes)

    def read_csv_input(self, csv_path, data_root):
        csv_data = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=';')
            for row in csv_reader:
                item = ListDataJpeg(row[0], row[1], os.path.join(data_root, row[0]))
                csv_data.append(item)
        return csv_data

    def read_csv_labels(self, csv_path):
        classes = []
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile)
            for row in csv_reader:
                classes.append(row[0])
        return classes

    def get_two_way_dict(self, classes):
        classes_dict = {}
        for i, item in enumerate(classes):
            classes_dict[item] = i
            classes_dict[i] = item
        return classes_dict


def default_loader(path, size=(176, 100)):
    return Image.open(path).convert('RGB').resize(size)


class VideoFolder(Dataset):

    def __init__(self, root, csv_file_input, csv_file_labels, clip_size,
                 nclips, step_size, is_val, transform=None,
                 loader=default_loader):
        self.dataset_object = JpegDataset(csv_file_input, csv_file_labels, root)

        self.csv_data = self.dataset_object.csv_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform = transform
        self.loader = loader

        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.is_val = is_val

    def __getitem__(self, index):
        item = self.csv_data[index]
        img_paths = self.get_frame_names(item.path)

        imgs = []
        for img_path in img_paths:
            img = self.loader(img_path)
            img = self.transform(img)
            imgs.append(torch.unsqueeze(img, 0))

        target_idx = self.classes_dict[item.label]

        # format data to torch
        data = torch.cat(imgs)
        data = data.permute(1, 0, 2, 3)
        return (data, target_idx)

    def __len__(self):
        return len(self.csv_data)

    def get_frame_names(self, path):
        frame_names = []
        for ext in IMG_EXTENSIONS:
            frame_names.extend(glob.glob(os.path.join(path, "*" + ext)))
        frame_names = list(sorted(frame_names))
        num_frames = len(frame_names)

        # set number of necessary frames
        if self.nclips > -1:
            num_frames_necessary = self.clip_size * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames

        # pick frames
        offset = 0
        if num_frames_necessary > num_frames:
            # pad last frame if video is shorter than necessary
            frame_names += [frame_names[-1]] * (num_frames_necessary - num_frames)
        elif num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset
            diff = (num_frames - num_frames_necessary)
            # Temporal augmentation
            if not self.is_val:
                offset = np.random.randint(0, diff)
        frame_names = frame_names[offset:num_frames_necessary +
                                  offset:self.step_size]
        return frame_names


class JesterDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, val_split=0.0, num_workers=4, split='train', clip_size=18, nclips=1, step_size=2):
        transform = transforms.Compose([
                    transforms.CenterCrop(84),
                    transforms.ToTensor(),
                    # transforms.Normalize(
                    #     mean=[0.485, 0.456, 0.406],
                    #     std=[0.229, 0.224, 0.225])
                    ])
        self.data_dir = data_dir
        self.clip_size = clip_size
        self.nclips = nclips
        self.step_size = step_size

        if split=='train':
            self.dataset = VideoFolder(
                self.data_dir,
                '/u/big/trainingdata/20BNJESTER/jester-v1-train.csv',
                '/u/big/trainingdata/20BNJESTER/jester-v1-labels.csv',
                clip_size=clip_size,
                nclips=nclips,
                step_size=step_size,
                is_val=False,
                transform=transform,
                loader=default_loader
            )
        elif split=='val':
            self.dataset = VideoFolder(
                self.data_dir,
                csv_file_input='/u/big/trainingdata/20BNJESTER/jester-v1-validation.csv',
                csv_file_labels='/u/big/trainingdata/20BNJESTER/jester-v1-labels.csv',
                clip_size=clip_size,
                nclips=nclips,
                step_size=step_size,
                is_val=True,
                transform=transform,
                loader=default_loader
            )
        else:
            self.dataset = VideoFolder(
                self.data_dir,
                csv_file_input='/u/big/trainingdata/20BNJESTER/jester-v1-validation.csv',
                csv_file_labels='/u/big/trainingdata/20BNJESTER/jester-v1-labels.csv',
                clip_size=clip_size,
                nclips=nclips,
                step_size=step_size,
                is_val=True,
                transform=transform,
                loader=default_loader
            ) 

        super().__init__(self.dataset, batch_size, shuffle, val_split, num_workers)

transform = transforms.Compose([
                        transforms.CenterCrop(84),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                        ])

                    
class TensorCenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        shape = img.size()
        h = shape[-2]
        w = shape[-1]
        th, tw = self.size
        i = int(round((h-th)/2.))
        j = int(round((w-tw)/2.))
        return img[:, :, :, i:i+th, j:j+tw]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class TensorNormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        if not self.inplace:
            tensor = tensor.clone()

        mean = torch.tensor(self.mean, dtype=torch.float32)#.expand_as(tensor)
        std = torch.tensor(self.std, dtype=torch.float32)#.expand_as(tensor)
        mean = mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(tensor)
        std = std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).expand_as(tensor)
        tensor.sub_(mean).div_(std)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

eval_transform = transforms.Compose([
    TensorCenterCrop(84),
    TensorNormalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
if __name__ == '__main__':
    transform = transforms.Compose([
                        transforms.CenterCrop(84),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
                        ])
    loader = VideoFolder(root="/u/big/trainingdata/20BNJESTER/20bn-jester-v1/",
                         csv_file_input="/u/big/trainingdata/20BNJESTER/jester-v1-train.csv",
                         csv_file_labels="/u/big/trainingdata/20BNJESTER/jester-v1-labels.csv",
                         clip_size=18,
                         nclips=1,
                         step_size=2,
                         is_val=False,
                         transform=transform,
                         loader=default_loader)
    # # data_item, target_idx = loader[0]
    # # save_images_for_debug("input_images", data_item.unsqueeze(0))

    train_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=10, shuffle=False,
        num_workers=5, pin_memory=True)
    train_loader = JesterDataLoader('/u/big/trainingdata/20BNJESTER/20bn-jester-v1/', 10)

    start = time.time()
    for i, a in enumerate(train_loader):
        print("Size --> {}, target --> {}".format(a[0].size(), a[1].type()))
        print("Max --> {}, Min --> {}".format(a[0].max(), a[0].min()))
        if i == 49:
            break
    print("Size --> {}".format(a[0].size()))
    print(time.time() - start)