"""Penn Treebank loading utilities compatible with the original TF example."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch


def _read_words(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").replace("\n", "<eos>").split()


def build_vocab(train_path: Path) -> Dict[str, int]:
    counter = Counter(_read_words(train_path))
    count_pairs = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    words = [word for word, _ in count_pairs]
    return {word: idx for idx, word in enumerate(words)}


def file_to_word_ids(path: Path, word_to_id: Dict[str, int]) -> List[int]:
    return [word_to_id[word] for word in _read_words(path)]


def ptb_raw_data(data_path: Union[str, Path]) -> Tuple[List[int], List[int], List[int], int]:
    root = Path(data_path)
    train_path = root / "ptb.train.txt"
    valid_path = root / "ptb.valid.txt"
    test_path = root / "ptb.test.txt"

    missing = [str(path) for path in (train_path, valid_path, test_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PTB files: " + ", ".join(missing))

    word_to_id = build_vocab(train_path)
    train = file_to_word_ids(train_path, word_to_id)
    valid = file_to_word_ids(valid_path, word_to_id)
    test = file_to_word_ids(test_path, word_to_id)
    return train, valid, test, len(word_to_id)


class PTBBatchedSplit:
    """Pre-batched view of a PTB split that can be reused across epochs.

    Layout matches the original ``ptb_iterator``: the token stream is reshaped
    into ``[batch_size, batch_len]`` row-major, then sliced along the time axis
    in fixed-size ``num_steps`` windows. The slicing order is identical to the
    original generator, so optimizer trajectories are bit-for-bit equivalent
    when training in FP32.
    """

    def __init__(
        self,
        raw_data: Iterable[int],
        batch_size: int,
        num_steps: int,
        device: torch.device,
        pin_memory: bool = False,
    ) -> None:
        if isinstance(raw_data, np.ndarray):
            raw = np.ascontiguousarray(raw_data, dtype=np.int64)
        else:
            raw = np.fromiter(raw_data, dtype=np.int64)
        data_len = int(raw.shape[0])
        batch_len = data_len // batch_size
        if batch_len <= 0:
            raise ValueError("batch_len <= 0, decrease batch_size")
        epoch_size = (batch_len - 1) // num_steps
        if epoch_size <= 0:
            raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

        layout = raw[: batch_size * batch_len].reshape(batch_size, batch_len)
        layout_tensor = torch.from_numpy(np.ascontiguousarray(layout))

        cuda_device = device if device.type == "cuda" else None
        if cuda_device is not None:
            layout_tensor = layout_tensor.to(cuda_device, non_blocking=False)
        elif pin_memory and torch.cuda.is_available():
            layout_tensor = layout_tensor.pin_memory()

        self.batch_size = batch_size
        self.num_steps = num_steps
        self.epoch_size = int(epoch_size)
        self.batch_len = int(batch_len)
        self.device = device
        self._layout = layout_tensor
        self._on_cuda = cuda_device is not None

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        layout = self._layout
        num_steps = self.num_steps
        device = self.device
        for i in range(self.epoch_size):
            start = i * num_steps
            end = start + num_steps
            x = layout[:, start:end]
            y = layout[:, start + 1 : end + 1]
            if self._on_cuda or x.device == device:
                yield x, y
            else:
                yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def ptb_iterator(
    raw_data: Iterable[int],
    batch_size: int,
    num_steps: int,
    device: torch.device,
    cache: Optional[PTBBatchedSplit] = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(input, target)`` batches matching the original PTB layout.

    If ``cache`` is provided it is reused; otherwise a fresh
    :class:`PTBBatchedSplit` is built for this single pass. Callers that
    iterate the same data many times (training over multiple epochs) should
    construct a :class:`PTBBatchedSplit` once and pass it in to avoid
    repeating the host-side reshape and ``host -> device`` copy on every
    epoch.
    """

    if cache is None:
        cache = PTBBatchedSplit(raw_data, batch_size, num_steps, device)
    elif (
        cache.batch_size != batch_size
        or cache.num_steps != num_steps
        or cache.device != device
    ):
        cache = PTBBatchedSplit(raw_data, batch_size, num_steps, device)
    yield from cache
