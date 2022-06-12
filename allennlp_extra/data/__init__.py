from allennlp_extra.data import data_loaders, dataset_readers


import math
from typing import List, Iterable, Tuple, Sequence, Optional
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers.batch_sampler import BatchSampler



@BatchSampler.register("my_sampler")
class MySampler(BatchSampler):

    def __init__(
        self,
        batch_size: int,
    ):
        self.batch_size = batch_size


    def get_batch_indices(self, instances: Sequence[Instance]) -> Iterable[List[int]]:
        indices = list(range(0, 3200)) + list(range(0, 3200)) + list(range(0, 3200)) + \
            list(range(3200, 4800)) + list(range(3200, 4800)) + list(range(3200, 4800)) + \
            list(range(4800, 5600)) + list(range(4800, 5600)) + list(range(4800, 5600)) + \
            list(range(5600, len(instances))) + list(range(5600, len(instances))) + list(range(5600, len(instances)))    # 50*3=150 steps, 151-225 steps ，226-281 ， 282-335
        batches = []
        for group in lazy_groups_of(indices, self.batch_size):
            batch_indices = list(group)
            batches.append(batch_indices)
        
        for batch in batches:
            yield batch


    def get_num_batches(self, instances: Sequence[Instance]) -> int:
        batch_count_float = (len(instances)*3 / self.batch_size)
        return math.ceil(batch_count_float)

    def get_batch_size(self) -> Optional[int]:
        return self.batch_size