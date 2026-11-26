"""Public, pinned Hub assets with resumable ranges and official object checksums."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import threading
import time
import urllib.request
from pathlib import Path


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    digest = hashlib.sha256() if entry['sha256'] else hashlib.sha1()
    if not entry['sha256']:
        digest.update(f'blob {entry["size"]}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == (entry['sha256'] or entry['git_blob'])


def fetch_one(url, dest, entry, workers):
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock_file = dest.with_name(dest.name + '.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if verify(dest, entry):
        print('Already verified:', entry['path'], flush=True)
        return
    size = entry['size']
    partial = dest.with_name(dest.name + '.partial')
    if size < 128 * 1024 * 1024:
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'huggingface_hub/0.36.2'})
                with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    else:
        progress = partial.with_name(partial.name + '.json')
        identity = {k: entry[k] for k in ('size', 'sha256', 'git_blob')}
        if partial.exists() and progress.exists():
            state = json.loads(progress.read_text())
            if state['identity'] != identity:
                raise ValueError('Partial asset identity changed')
            done = set(state['done'])
        else:
            done = set()
            with partial.open('wb') as f:
                f.truncate(size)
        chunk_size = 64 * 1024 * 1024
        count = (size + chunk_size - 1) // chunk_size
        fd = os.open(partial, os.O_RDWR)
        mutex = threading.Lock()
        def save_progress():
            tmp = progress.with_name(progress.name + '.tmp')
            tmp.write_text(json.dumps({'identity': identity, 'done': sorted(done)}))
            tmp.replace(progress)
        save_progress()
        def download(index):
            if index in done:
                return
            start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
            for attempt in range(5):
                try:
                    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}',
                                                                  'User-Agent': 'huggingface_hub/0.36.2'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Unexpected response range')
                        offset = start
                        while block := response.read(1024 * 1024):
                            if offset + len(block) > end + 1:
                                raise ValueError('Oversized range')
                            written = os.pwrite(fd, block, offset)
                            if written != len(block):
                                raise IOError('Short file write')
                            offset += written
                        if offset != end + 1:
                            raise IOError('Incomplete response range')
                    with mutex:
                        done.add(index)
                        save_progress()
                        if len(done) % 4 == 0 or len(done) == count:
                            print(f'{entry["path"]}: {len(done)}/{count} chunks', flush=True)
                    return
                except Exception as error:
                    print(f'{entry["path"]}: range {index}, retry {attempt + 1}, {type(error).__name__}', flush=True)
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(download, range(count)))
            os.fsync(fd)
        finally:
            os.close(fd)
    if not verify(partial, entry):
        raise ValueError(f'Official object checksum mismatch: {entry["path"]}')
    partial.replace(dest)
    print('Verified:', entry['path'], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        dest = (root / entry['path']).resolve()
        if not dest.is_relative_to(root):
            raise ValueError('Asset path escapes destination')
        url = f'{a.endpoint.rstrip("/")}/{manifest["repo"]}/resolve/{manifest["revision"]}/{entry["path"]}?download=true'
        fetch_one(url, dest, entry, a.workers)
    (root / 'origin.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('ALL_ASSETS_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
