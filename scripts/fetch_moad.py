"""Recover only official split biounits from the public Binding MOAD archive.

HTTP ranges avoid downloading the two full 3 GB archives. Each extracted member
is validated against its ZIP CRC32; full-archive MD5 is metadata, not recomputed.
"""
import concurrent.futures as cf
import hashlib
import json
import struct
import time
import zlib
from pathlib import Path
import requests

from tasks import ROOT, CODE_ROOT
SPLIT = CODE_ROOT/'cohorts/moad_test.txt'
ARCHIVES = {'a': 3315455002, 'b': 2940114726}

def get_range(url, start, end):
    for attempt in range(4):
        try:
            # Vary URL as well as Range to avoid proxies reusing another range.
            r = requests.get(url+f'&range_start={start}&range_end={end}', headers={'Range': f'bytes={start}-{end}'}, timeout=(20, 60))
            r.raise_for_status()
            expected = f'bytes {start}-{end}/'
            if r.status_code != 206 or not r.headers.get('Content-Range','').startswith(expected):
                raise RuntimeError(f'range unsupported/mismatched: {r.status_code} {r.headers.get("Content-Range")}')
            if len(r.content) != end-start+1:
                raise RuntimeError('short range')
            return r.content
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt+1)

def index_archive(part, size):
    url = f'https://zenodo.org/records/17702010/files/every_part_{part}.zip?download=1'
    tail_file = ROOT/'data'/f'moad_zip_{part}_tail.bin'
    tail = tail_file.read_bytes() if tail_file.exists() else get_range(url, size-65536, size-1)
    eocd = tail.rfind(b'PK\x05\x06')
    _,_,_,_,entries,cdsize,cdoffset,_ = struct.unpack_from('<4s4H2IH',tail,eocd)
    cache = ROOT/'data'/f'moad_zip_{part}_central.bin'
    central = cache.read_bytes() if cache.exists() else get_range(url,cdoffset,cdoffset+cdsize-1)
    cache.write_bytes(central)
    found=[]; offset=0
    while offset < len(central):
        h = struct.unpack_from('<4s6H3I5H2I',central,offset)
        if h[0] != b'PK\x01\x02':
            raise ValueError('invalid central directory')
        name = central[offset+46:offset+46+h[10]].decode('utf8')
        found.append(dict(name=name, crc32=h[7], compressed_size=h[8], size=h[9], method=h[4], header_offset=h[16], filename_len=h[10], extra_len=h[11], archive=part, url=url))
        offset += 46+h[10]+h[11]+h[12]
    assert len(found)==entries
    print(json.dumps(dict(archive=part, entries=entries, central_bytes=cdsize)),flush=True)
    return found

def extract(item):
    name = Path(item['name']).name
    dest=ROOT/'data'/'moad_raw'/name
    try:
        if not dest.exists():
            start=item['header_offset']
            end=start+30+item['filename_len']+item['extra_len']+item['compressed_size']+1024-1
            blob=get_range(item['url'],start,end)
            header=struct.unpack_from('<4s5H3I2H',blob,0)
            assert header[0]==b'PK\x03\x04'
            offset=30+header[9]+header[10]
            content=blob[offset:offset+item['compressed_size']]
            if item['method']==8:
                content=zlib.decompress(content,-15)
            elif item['method']!=0:
                raise ValueError('unsupported ZIP compression')
            assert len(content)==item['size']
            assert zlib.crc32(content)==item['crc32']
            dest.write_bytes(content)
        content=dest.read_bytes()
        assert zlib.crc32(content)==item['crc32']
        return dict(file=name,status='verified',sha256=hashlib.sha256(content).hexdigest(),**item)
    except Exception as e:
        return dict(file=name,status='failed',error=repr(e),**item)

if __name__=='__main__':
    (ROOT/'data'/'moad_raw').mkdir(exist_ok=True)
    ids=SPLIT.read_text().strip().split(',')
    (ROOT/'data'/'official_moad_test_ids.json').write_text(json.dumps(ids,indent=2))
    wanted={x.split('_')[0].lower().replace('-bio','.bio') for x in ids}
    entries=[]
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        for result in ex.map(lambda x:index_archive(*x),ARCHIVES.items()): entries+=result
    selected=[x for x in entries if Path(x['name']).name.lower() in wanted]
    found={Path(x['name']).name.lower() for x in selected}
    print(json.dumps(dict(split_cases=len(ids),required_biounits=len(wanted),found=len(found),missing=sorted(wanted-found))),flush=True)
    report=dict(source_record='https://zenodo.org/records/17702010',official_split=str(SPLIT),split_cases=len(ids),required_biounits=len(wanted),missing=sorted(wanted-found),results=[])
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for fut in cf.as_completed([ex.submit(extract,x) for x in selected]):
            result=fut.result();report['results'].append(result)
            print(json.dumps({k:result[k] for k in ['file','status']}),flush=True)
            (ROOT/'results'/'moad_download_manifest.json').write_text(json.dumps(report,indent=2))
    if report['missing'] or any(row['status'] != 'verified' for row in report['results']):
        raise RuntimeError('MOAD download is incomplete; inspect the manifest and retry before preparation')
