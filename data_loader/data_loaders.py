from glob import glob
import os

from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset

# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        

class JESTER:
    classes = {
        "Doing other things": 0,
        "Drumming Fingers": 1,
        "No gesture": 2,
        "Pulling Hand In": 3,
        "Pulling Two Fingers In":4,
        "Pushing Hand Away":5,
        "Pushing Two Fingers Away":6,
        "Rolling Hand Backward":7,
        "Rolling Hand Forward":8,
        "Shaking Hand":9,
        "Sliding Two Fingers Down":10,
        'Sliding Two Fingers Left':11,
        "Sliding Two Fingers Right":12,
        "Sliding Two Fingers Up":13,
        "Stop Sign":14,
        "Swiping Down":15,
        "Swiping Left":16,
        "Swiping Right":17,
        'Swiping Up':18,
        'Thumb Down':19,
        'Thumb Up':20,
        'Turning Hand Clockwise':21,
        'Turning Hand Counterclockwise':22,
        'Zooming In With Full Hand':23,
        'Zooming In With Two Fingers':24,
        'Zooming Out With Full Hand':25,
        'Zooming Out With Two Fingers':26
    }

    image_dir = '20bn-jester-v1'
    train_label_file = 'jester-v1-train.csv'
    val_label_file = 'jester-v1-validation.csv'
    test_label_file = 'jester-v1-test.csv'


    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.image_dir = os.path.join(self.root, self.image_dir)
        

        
        if self.split == 'train':
            self.file_name = os.path.join(self.root, self.train_label_file)
        elif self.split == 'train':
            self.file_name = os.path.join(self.root, self.val_label_file)
        else:
            self.file_name = os.path.join(self.root, self.test_label_file)

        self.fnames = []
        self.targets = []
        self.parse_fnames(self.file_name)

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        images = load_directory(self.fnames[idx])
        target = self.targets[idx]
        if self.transform is not None:
            images = self.transform(images)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    def parse_fnames(self, fname):
        with open(fname, 'r') as f:
            for line in f:
                _id, action = line.strip().split(';')
                directory = [os.path.join(self.image_dir, _id, fname) for fname in os.listdir(os.path.join(self.image_dir, _id))]
                self.fnames.append(directory)                
                self.targets.append(self.classes[action])
        
    def load_directory(directory):
        images = [imread(fname) for fname in directory]
        return images



class JesterDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, val_split=0.0, num_workers=4, split='train'):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = JESTER(self.data_dir, split=split, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, val_split, num_workers)
        