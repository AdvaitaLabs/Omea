"""Read official Figshare ZIP via byte ranges; extract only inference weights."""
import io
import json
import time
import zipfile
from pathlib import Path
import requests

from benchmark_io import ROOT


class RemoteFile(io.RawIOBase):
    def __init__(self):
        self.session = requests.Session()
        url = 'https://ndownloader.figshare.com/files/68203198?download=1&nocache=' + str(time.time_ns())
        r = self.session.get(url, headers={'Range': 'bytes=-65536'}, timeout=90)
        r.raise_for_status()
        assert r.status_code == 206
        self.url = r.url
        self.size = int(r.headers['Content-Range'].split('/')[-1])
        self.pos = 0
        self.tail = r.content

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else self.size if whence == 2 else 0)
        return self.pos
    def read(self, n=-1):
        if n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if not n: return b''
        if self.pos >= self.size - len(self.tail):
            out = self.tail[self.pos - (self.size - len(self.tail)):][:n]
        else:
            end = self.pos + n - 1
            for attempt in range(4):
                try:
                    # Figshare signs each redirect for just 10 seconds. Obtain
                    # a new redirect for every range, not a saved S3 URL.
                    url = 'https://ndownloader.figshare.com/files/68203198?download=1&nocache=' + str(time.time_ns())
                    r = self.session.get(url, headers={'Range': f'bytes={self.pos}-{end}'}, timeout=180)
                    r.raise_for_status()
                    assert r.status_code == 206 and r.headers['Content-Range'] == f'bytes {self.pos}-{end}/{self.size}'
                    out = r.content
                    assert len(out) == n
                    break
                except Exception:
                    if attempt == 3: raise
        self.pos += len(out)
        return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--extract', action='store_true'); a = p.parse_args()
    with zipfile.ZipFile(RemoteFile()) as z:
        files = [{'name': i.filename, 'bytes': i.file_size, 'compressed': i.compress_size, 'crc32': i.CRC} for i in z.infolist()]
        (ROOT / 'checkpoints/neat_zip_index.json').write_text(json.dumps(files, indent=2))
        for i in files: print(i, flush=True)
        if a.extract:
            dest = ROOT / 'checkpoints/neat_pocket'
            for info in z.infolist():
                name = info.filename
                if not ('neat_spindr/' in name or 'bond_predictor/bp_spindr/' in name): continue
                if not (name.endswith('.ckpt') or name.endswith('.yaml')): continue
                path = dest / name
                assert path.resolve().is_relative_to(dest.resolve())
                if path.exists() and path.stat().st_size == info.file_size: continue
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix('.part')
                print('Extracting', name, info.file_size, flush=True)
                with z.open(info) as src, tmp.open('wb') as out:
                    while chunk := src.read(8 * 1024 * 1024):
                        out.write(chunk)
                assert tmp.stat().st_size == info.file_size
                tmp.replace(path)
                print('Finished', name, flush=True)
