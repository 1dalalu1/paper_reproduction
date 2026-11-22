"""下载 HuggingFace 上的 RouterBench 原始数据文件。

用法：
    python scripts/download_data.py                 # 自动探测本机 HTTP 代理
    python scripts/download_data.py --no-proxy      # 直连（境外网络可达时）
    python scripts/download_data.py --proxy http://127.0.0.1:7897
    python scripts/download_data.py --files routerbench_raw.pkl --dest data/raw

为什么用 Python 的 urllib 而不是 curl / Invoke-WebRequest：
Windows 沙箱里 curl 走 Schannel 会报 `SEC_E_NO_CREDENTIALS`，Invoke-WebRequest 直接断连；
Python 用自带的 OpenSSL，配合显式 HTTP 代理可以正常出网。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import _bootstrap

_bootstrap.setup()

from routerbench.config import load_config  # noqa: E402

BASE = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/"
DEFAULT_FILES = ("routerbench_0shot.pkl", "routerbench_5shot.pkl")
DEFAULT_PROXIES = ("http://127.0.0.1:7897", "http://127.0.0.1:7890", "http://127.0.0.1:10809")


def _probe(proxy: str | None, timeout: float = 8.0) -> bool:
    """用一个小文件探一下代理是否可用。"""
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    opener = urllib.request.build_opener(*handlers)
    try:
        with opener.open(BASE + "README.md", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def download(url: str, dst: Path, proxy: str | None, retries: int = 5) -> None:
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    opener = urllib.request.build_opener(*handlers)
    for attempt in range(retries):
        try:
            t0 = time.time()
            req = urllib.request.Request(url)
            with opener.open(req, timeout=120) as r, open(f"{dst}.part", "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total and got // (32 << 20) != (got - len(chunk)) // (32 << 20):
                        speed = got / 1e6 / max(time.time() - t0, 1e-6)
                        print(f"    {got/1e6:6.1f}/{total/1e6:.1f} MB  {speed:5.1f} MB/s", flush=True)
            if total and got != total:
                raise IOError(f"下载不完整 {got}/{total} bytes")
            os.replace(f"{dst}.part", dst)
            print(f"  OK  {dst.name}  {dst.stat().st_size/1e6:.2f} MB  {time.time()-t0:.1f}s")
            return
        except Exception as e:  # noqa: BLE001 - 网络异常种类很多，统一重试
            print(f"  retry {attempt + 1}/{retries}  {type(e).__name__}: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    raise SystemExit(f"下载失败：{url}")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 RouterBench 原始数据")
    ap.add_argument("--dest", default=None, help="下载目录（默认取 config 的 data/raw）")
    ap.add_argument("--proxy", default=None, help="HTTP 代理，例如 http://127.0.0.1:7897")
    ap.add_argument("--no-proxy", action="store_true", help="不使用代理")
    ap.add_argument("--files", nargs="*", default=list(DEFAULT_FILES))
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="已存在也重下")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dest = Path(args.dest) if args.dest else cfg.raw_dir
    dest.mkdir(parents=True, exist_ok=True)

    todo = [f for f in args.files if args.force or not (dest / f).exists()]
    skipped = [f for f in args.files if f not in todo]
    for f in skipped:
        print(f"已存在，跳过：{dest / f}")

    if not todo:
        return 0

    proxy: str | None = None
    if not args.no_proxy:
        candidates = [args.proxy] if args.proxy else [
            os.environ.get("HTTPS_PROXY"), os.environ.get("https_proxy"), *DEFAULT_PROXIES
        ]
        for cand in candidates:
            if cand and _probe(cand):
                proxy = cand
                break
        print(f"代理：{proxy or '直连'}")

    for fn in todo:
        print(f"下载 {fn} ...")
        download(BASE + fn, dest / fn, proxy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
