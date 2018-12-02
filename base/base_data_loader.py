import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, valation_split, num_workers, collate_fn=default_collate):
        self.valation_split = valation_split
        self.shuffle = shuffle
        
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.val_sampler = self._split_sampler(self.valation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0) 
        np.random.shuffle(idx_full)

        len_val = int(self.n_samples * split)

        val_idx = idx_full[0:len_val]
        train_idx = np.delete(idx_full, np.arange(0, len_val))
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, val_sampler
        
    def split_valation(self):
        if self.val_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.val_sampler, **self.init_kwargs)
    
