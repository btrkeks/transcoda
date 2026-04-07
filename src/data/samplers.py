import random
from collections.abc import Iterator, Sized

import numpy as np
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler[list[int]]):
    """
    A batch sampler that groups samples of similar sizes into buckets to minimize padding.

    This sampler reduces GPU memory spikes by ensuring that batches contain images of similar
    dimensions, which results in more consistent padded tensor sizes across batches.

    Args:
        data_source: The dataset to sample from. Must be Sized.
        sample_sizes: A list of sizes (e.g., image area) for each sample in the dataset.
        batch_size: The target number of samples per batch.
        drop_last: If True, the sampler will drop the last batch if its size is less than batch_size.
        bucket_width_factor: A factor to control the size of buckets. Larger values mean more mixing
                            but potentially more padding. Default is 10, meaning buckets contain
                            batch_size * 10 samples.
    """

    def __init__(
        self,
        data_source: Sized,
        sample_sizes: list[int | float],
        batch_size: int,
        drop_last: bool,
        bucket_width_factor: int = 10,
    ) -> None:
        if not isinstance(data_source, Sized):
            raise TypeError("data_source should be a Sized object.")
        super().__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 1. Create buckets based on sample sizes
        indices = np.argsort(sample_sizes)
        bucket_size = self.batch_size * bucket_width_factor
        self.buckets = [
            indices[i : i + bucket_size].tolist() for i in range(0, len(indices), bucket_size)
        ]
        self.num_batches = self._calculate_num_batches()

    def _calculate_num_batches(self) -> int:
        num_batches = 0
        for bucket in self.buckets:
            num_batches += len(bucket) // self.batch_size
            if not self.drop_last and len(bucket) % self.batch_size > 0:
                num_batches += 1
        return num_batches

    def __iter__(self) -> Iterator[list[int]]:
        # Shuffle the buckets themselves to maintain randomness across epochs
        random.shuffle(self.buckets)

        for bucket in self.buckets:
            # Shuffle indices within each bucket
            random.shuffle(bucket)

            # Yield batches from the current bucket
            for i in range(0, len(bucket), self.batch_size):
                batch_indices = bucket[i : i + self.batch_size]
                if len(batch_indices) < self.batch_size and self.drop_last:
                    continue
                yield batch_indices

    def __len__(self) -> int:
        return self.num_batches
