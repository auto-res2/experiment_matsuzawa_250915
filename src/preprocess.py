"""preprocess.py – download & prepare OGBN-papers400M / OpenRoad datasets."""
from __future__ import annotations
import argparse, tarfile, urllib.request, json
from pathlib import Path
from typing import Tuple, Dict

import dgl, torch, numpy as np, zstandard as zstd
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

OPENROAD_URL_DEFAULT = "https://zenodo.org/record/11030517/files/openroad_nyc.tar.zst"


# ---------------------------------------------------------------------------
# Helper I/O functions
# ---------------------------------------------------------------------------

def _download(url: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    with urllib.request.urlopen(url) as resp, open(out_path, "wb") as out_f:
        total = int(resp.info()["Content-Length"])
        pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {url}")
        while True:
            chunk = resp.read(8 << 20)  # 8 MB
            if not chunk:
                break
            out_f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()


def _extract_zst(zst_file: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(zst_file, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                tar.extractall(path=out_dir)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_ogbn_papers400m(root: str, use_subset: bool) -> Tuple[dgl.DGLGraph, Dict[str, torch.Tensor]]:
    dataset = DglNodePropPredDataset("ogbn-papers400m", root)
    g, labels = dataset[0]
    g.ndata["label"] = labels.squeeze()
    if use_subset:
        idx = torch.arange(50_000)
        g = dgl.node_subgraph(g, idx)
    return g, dataset.get_idx_split()


def _prepare_openroad(url: str, root: str):
    out = Path(root) / "openroad_nyc.tar.zst"
    _download(url, out)
    _extract_zst(out, Path(root) / "openroad_nyc")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))  # noqa: F821 – yaml comes from pyyaml, installed per pyproject

    # OGBN
    g, _ = load_ogbn_papers400m(cfg["dataset"]["root"], cfg["dataset"].get("use_subset", False))
    meta = {"num_nodes": int(g.num_nodes()), "num_edges": int(g.num_edges())}
    Path(cfg["dataset"]["root"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["dataset"]["root"]) / "meta.json").write_text(json.dumps(meta))

    # OpenRoad if requested
    if cfg.get("openroad") is not None:
        _prepare_openroad(cfg["openroad"].get("url", OPENROAD_URL_DEFAULT), cfg["openroad"]["root"])


if __name__ == "__main__":
    import yaml  # noqa – local import to avoid top-level dependency when unused
    main()
