"""Tokenizer training and dataloader construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, random_split

from transformer_lib.config import SPECIAL_TOKENS, UNK_TOKEN, Config
from transformer_lib.data.bilingual import BilingualDataset


def sentence_iterator(dataset: Any, language: str) -> Iterator[str]:
    for item in dataset:
        yield item["translation"][language]


def get_or_build_tokenizer(config: Config, dataset: Any, lang: str) -> Tokenizer:
    tokenizer_path = config.tokenizer_path(lang)
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS, min_frequency=2)
    tokenizer.train_from_iterator(sentence_iterator(dataset, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def analyze_max_seq_lengths(
    dataset: Any,
    tokenizer_src: Tokenizer,
    tokenizer_tgt: Tokenizer,
    src_lang: str,
    tgt_lang: str,
) -> tuple[int, int]:
    max_src, max_tgt = 0, 0
    for item in dataset:
        max_src = max(max_src, len(tokenizer_src.encode(item["translation"][src_lang]).ids))
        max_tgt = max(max_tgt, len(tokenizer_tgt.encode(item["translation"][tgt_lang]).ids))
    return max_src, max_tgt


def recommended_seq_len(max_src: int, max_tgt: int, buffer: int = 10) -> int:
    return max(max_src + 2, max_tgt + 1) + buffer


def get_translation_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, Tokenizer, Tokenizer]:
    subset = f"{config.data.lang_src}-{config.data.lang_tgt}"
    ds_raw = load_dataset(config.data.dataset_name, subset, split="train")

    if config.data.max_train_samples is not None:
        n = min(config.data.max_train_samples, len(ds_raw))
        ds_raw = ds_raw.select(range(n))
        print(f"Using subset of {n} training samples (max_train_samples)")

    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config.data.lang_src)
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config.data.lang_tgt)

    max_src, max_tgt = analyze_max_seq_lengths(
        ds_raw, tokenizer_src, tokenizer_tgt, config.data.lang_src, config.data.lang_tgt
    )
    print(f"Max tokenized source length: {max_src}")
    print(f"Max tokenized target length: {max_tgt}")
    rec = recommended_seq_len(max_src, max_tgt)
    if config.model.seq_len < rec - 10:
        print(
            f"Warning: config.model.seq_len={config.model.seq_len} may be too small. "
            f"Recommended >= {rec}"
        )

    train_size = int(config.data.train_split_ratio * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_raw, val_raw = random_split(ds_raw, [train_size, val_size])

    seq_len = config.model.seq_len
    truncate = config.data.truncate_long
    train_ds = BilingualDataset(
        train_raw, tokenizer_src, tokenizer_tgt,
        config.data.lang_src, config.data.lang_tgt, seq_len,
        truncate_long=truncate,
    )
    val_ds = BilingualDataset(
        val_raw, tokenizer_src, tokenizer_tgt,
        config.data.lang_src, config.data.lang_tgt, seq_len,
        truncate_long=truncate,
    )

    use_cuda = __import__("torch").cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
    )
    return train_loader, val_loader, tokenizer_src, tokenizer_tgt
