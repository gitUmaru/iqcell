import torch
from torch.utils.data import Sampler



class RandomSampler(Sampler):
    def __init__(self, data_source, replacement=True, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples if num_samples is not None else len(data_source)

    def __iter__(self):
        return iter(torch.randint(high=len(self.data_source), size=(self.num_samples,), dtype=torch.int64).tolist())

    def __len__(self):
        return self.num_samples

class IqSampler(Sampler[int]):
    def __init__(self, data_source, replacement  = False,
                    num_samples = None, generator=None):
            self.data_source = data_source
            self.replacement = replacement
            self.num_samples = num_samples if num_samples else len(data_source)
            self.generator = generator

    def __iter__(self):
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples