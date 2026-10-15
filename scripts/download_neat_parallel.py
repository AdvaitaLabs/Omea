"""Resume parallel public Figshare byte ranges for the SPINDR checkpoint."""
import concurrent.futures as cf
import json
import struct
import time
import zipfile
import zlib
import threading
from pathlib import Path
import requests
from download_neat_weights import RemoteFile, ROOT

NAME = 'trained_models/neat_spindr/checkpoints/periodic-epoch=4999.ckpt'
DEST = ROOT / 'checkpoints/neat_pocket' / NAME
PARTS = ROOT / 'checkpoints/neat_spindr_chunks'
META = PARTS / 'member.json'


def main():
    PARTS.mkdir(exist_ok=True)
    if META.exists():
        meta = json.loads(META.read_text())
    else:
        remote = RemoteFile()
        with zipfile.ZipFile(remote) as z:
            info = z.getinfo(NAME)
            remote.seek(info.header_offset)
            header = remote.read(30)
            fields = struct.unpack('<4s5H3I2H', header)
            assert fields[0] == b'PK\x03\x04'
            meta = dict(start=info.header_offset+30+fields[-2]+fields[-1],
                        compressed_bytes=info.compress_size, bytes=info.file_size,
                        crc32=info.CRC, method=info.compress_type, archive_bytes=remote.size)
            META.write_text(json.dumps(meta, indent=2))
    assert meta['method'] == 8
    chunk = 16 * 1024 * 1024
    ranges = [(i, start, min(chunk, meta['compressed_bytes']-start))
              for i, start in enumerate(range(0, meta['compressed_bytes'], chunk))]
    thread_local=threading.local()
    def fetch(row):
        i, offset, length = row; path = PARTS / f'{i:04d}.bin'
        if path.exists() and path.stat().st_size == length: return i
        start = meta['start']+offset; end = start+length-1
        for attempt in range(5):
            try:
                if not hasattr(thread_local,'session'):thread_local.session=requests.Session()
                url = 'https://api.figshare.com/v2/file/download/68203198?nocache='+str(time.time_ns())
                r = thread_local.session.get(url, headers={'Range':f'bytes={start}-{end}'}, timeout=(30,180))
                r.raise_for_status()
                assert r.status_code == 206
                assert r.headers['Content-Range'] == f"bytes {start}-{end}/{meta['archive_bytes']}"
                assert len(r.content) == length
                tmp = path.with_suffix('.part'); tmp.write_bytes(r.content); tmp.replace(path)
                return i
            except Exception as e:
                print(json.dumps(dict(chunk=i,retry=attempt+1,error_type=type(e).__name__)),flush=True)
                if attempt == 4: raise
                time.sleep(min(2**attempt,8))
    began = time.monotonic()
    failures=[]
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, row) for row in ranges]
        for i, future in enumerate(cf.as_completed(futures), 1):
            try:future.result()
            except Exception as e:failures.append(type(e).__name__)
            if i % 5 == 0 or i == len(ranges):
                print(json.dumps(dict(completed=i,total=len(ranges),seconds=round(time.monotonic()-began,1))),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} chunks failed; completed chunks retained for resume')
    dec = zlib.decompressobj(-15); crc=0; written=0
    DEST.parent.mkdir(parents=True,exist_ok=True);tmp=DEST.with_suffix('.parallel.part')
    with tmp.open('wb') as f:
        for i, _, _ in ranges:
            data = dec.decompress((PARTS/f'{i:04d}.bin').read_bytes())
            f.write(data); crc=zlib.crc32(data,crc);written+=len(data)
        data=dec.flush();f.write(data);crc=zlib.crc32(data,crc);written+=len(data)
    assert dec.eof and written==meta['bytes'] and crc==meta['crc32']
    tmp.replace(DEST)
    print(json.dumps(dict(status='complete',path=str(DEST),bytes=written,crc32=crc,seconds=time.monotonic()-began)),flush=True)


if __name__=='__main__': main()
