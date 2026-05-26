"""Dataloaders for distillation (same tokenizers as teacher)."""

from __future__ import annotations

import logging

from datasets import Dataset, load_dataset
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, random_split

from distil.config import DistilRunConfig
from transformer_lib.config import Config as TeacherConfig
from transformer_lib.data.bilingual import BilingualDataset
from transformer_lib.data.tokenization import get_or_build_tokenizer

logger = logging.getLogger(__name__)


def word_count(text: str) -> int:
    return len(text.split())


def filter_by_word_count(
    ds: Dataset,
    src_lang: str,
    tgt_lang: str,
    min_src_words: int | None,
    max_src_words: int | None,
    max_tgt_words: int | None,
) -> Dataset:
    if min_src_words is None and max_src_words is None and max_tgt_words is None:
        return ds

    before = len(ds)

    def keep(example: dict) -> bool:
        src = example["translation"][src_lang]
        tgt = example["translation"][tgt_lang]
        sw, tw = word_count(src), word_count(tgt)
        if min_src_words is not None and sw < min_src_words:
            return False
        if max_src_words is not None and sw > max_src_words:
            return False
        if max_tgt_words is not None and tw > max_tgt_words:
            return False
        return True

    ds = ds.filter(keep)
    logger.info(
        "Word-count filter (src %s–%s words, tgt ≤%s): %d → %d pairs",
        min_src_words or "any",
        max_src_words or "any",
        max_tgt_words or "any",
        before,
        len(ds),
    )
    return ds


def get_distil_dataloaders(
    distil_cfg: DistilRunConfig,
    teacher_cfg: TeacherConfig,
) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    subset = f"{teacher_cfg.data.lang_src}-{teacher_cfg.data.lang_tgt}"
    ds_raw = load_dataset(distil_cfg.data.dataset_name, subset, split="train")

    if distil_cfg.data.max_train_samples is not None:
        n = min(distil_cfg.data.max_train_samples, len(ds_raw))
        ds_raw = ds_raw.select(range(n))

    src, tgt = teacher_cfg.data.lang_src, teacher_cfg.data.lang_tgt
    ddata = distil_cfg.data
    ds_raw = filter_by_word_count(
        ds_raw,
        src,
        tgt,
        ddata.min_src_words,
        ddata.max_src_words,
        ddata.max_tgt_words,
    )

    tokenizer_src = get_or_build_tokenizer(teacher_cfg, ds_raw, src)
    tokenizer_tgt = get_or_build_tokenizer(teacher_cfg, ds_raw, tgt)

    train_size = int(distil_cfg.data.train_split_ratio * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

    seq_len = distil_cfg.student.seq_len
    truncate = distil_cfg.data.truncate_long

    train_ds = BilingualDataset(train_raw, tokenizer_src, tokenizer_tgt, src, tgt, seq_len, truncate)
    val_ds = BilingualDataset(val_raw, tokenizer_src, tokenizer_tgt, src, tgt, seq_len, truncate)

    use_cuda = __import__("torch").cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=distil_cfg.data.batch_size,
        shuffle=True,
        num_workers=distil_cfg.data.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt
