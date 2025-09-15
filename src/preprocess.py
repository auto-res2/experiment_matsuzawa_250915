"""Data acquisition & preprocessing pipeline for ORION experiments"""
from __future__ import annotations

import json
import os
import random
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import datasets
from datasets import DatasetDict, load_dataset
from tqdm.auto import tqdm

DATA_ROOT = Path("data")
CUSTOM_TOKEN_LIST_URL = "https://raw.githubusercontent.com/iamcal/emoji-data/master/emoji.json"


def _normalise(txt: str) -> str:
    return unicodedata.normalize("NFC", txt).strip().lower()


def _download_token_list() -> List[str]:
    import requests

    r = requests.get(CUSTOM_TOKEN_LIST_URL, timeout=30)
    r.raise_for_status()
    return [bytes.fromhex(e["unified"].replace("-", "")).decode("utf-8") for e in r.json()]


def _inject_tokens(
    msg_pairs: List[Tuple[str, str]], interval: int, n_min: int, n_max: int, token_pool: List[str], seed: int
) -> List[Tuple[str, str]]:
    random.seed(seed)
    out: List[Tuple[str, str]] = []
    for i, (m, r) in enumerate(msg_pairs):
        if (i + 1) % interval == 0:
            n_new = random.randint(n_min, n_max)
            r = f"{r} {' '.join(random.sample(token_pool, n_new))}"
        out.append((m, r))
    return out


def prepare_red_whatsapp(cfg: Dict[str, Any], seed: int, smoke: bool = False) -> DatasetDict:
    DATA_ROOT.mkdir(exist_ok=True)
    print("▶ Preparing Reddit / WhatsApp corpus …")
    redd = load_dataset(cfg["datasets"]["reddit"]["hf_name"], split=cfg["datasets"]["reddit"]["split"], token=os.getenv("HF_TOKEN"))
    wapp = load_dataset(cfg["datasets"]["whatsapp"]["hf_name"], split=cfg["datasets"]["whatsapp"]["split"], token=os.getenv("HF_TOKEN"))

    pairs: List[Tuple[str, str]] = []
    for rec in redd:
        if rec.get("body"):
            pairs.append((_normalise(rec["body"]), _normalise(rec["body"])) )
            if smoke and len(pairs) > 5_000:
                break
    for rec in wapp:
        if rec.get("text"):
            pairs.append((_normalise(rec["text"]), _normalise(rec["text"])) )
            if smoke and len(pairs) > 10_000:
                break

    pairs.sort(key=lambda x: len(x[0]))  # deterministic order
    pool = _download_token_list()
    pairs = _inject_tokens(pairs, cfg["token_injection"]["interval"], cfg["token_injection"]["n_min"], cfg["token_injection"]["n_max"], pool, seed)

    msgs, replies = zip(*pairs)
    ds = datasets.Dataset.from_dict({"msg": list(msgs), "reply": list(replies)})
    n = len(ds)
    tr_end, val_end = int(0.8 * n), int(0.9 * n)
    return DatasetDict(train=ds.select(range(tr_end)), validation=ds.select(range(tr_end, val_end)), test=ds.select(range(val_end, n)))


def load_malware_prompts(cfg: Dict[str, Any]):
    m = cfg["datasets"]["malware_sniffer"]
    return load_dataset(m["hf_name"], m.get("subset"), split=m["split"], token=os.getenv("HF_TOKEN"))


def load_emoji_chat(cfg: Dict[str, Any]):
    m = cfg["datasets"]["emoji_chat"]
    return load_dataset(m["hf_name"], split=m["split"], token=os.getenv("HF_TOKEN"))

__all__ = ["prepare_red_whatsapp", "load_malware_prompts", "load_emoji_chat"]