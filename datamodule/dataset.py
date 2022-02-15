from torch.utils.data import Dataset


class MTLDataset(Dataset):
    """Combines multiple datasets into a single Multi-Task dataset.
    """

    def __init__(self, datasets):
        """

        :param datasets:
        """
        self.datasets = datasets
        self.dataset_lengths = [len(ds) for ds in datasets]
        self.num_datasets = len(datasets)

    def __getitem__(self, indices):
        """

        :param indices:
        :return:
        """
        dataset_idx, sample_idx = indices
        return self.datasets[dataset_idx][sample_idx]

    def __len__(self):
        return self.dataset_lengths
