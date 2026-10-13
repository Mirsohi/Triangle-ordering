"""Datasets, downloads and preprocessing for pc_eval.

Parsing uses the artifact's own line parser `dove.graph._edge_stream` (identical rules by construction);
preprocessing is a vectorised equivalent of `dove.graph.graph_from_edges` (tested for identity in
tests/test_ports.py): drop self-loops, (min,max), deduplicate, remap vertices that occur in a remaining edge to
0..n-1 in increasing original id, sort edges.
"""
import csv
import gzip
import hashlib
import itertools
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

SNAP = 'https://snap.stanford.edu/data'
USER_AGENT = 'pc_eval/1.0'

# Pre-registered held-out graphs (PREREGISTRATION.md, Section 3.1), in processing order.
HELDOUT = [
    dict(name='Email-Eu-core', label='Email-Eu-core', url=f'{SNAP}/email-Eu-core.txt.gz', member=None),
    dict(name='CollegeMsg', label='CollegeMsg', url=f'{SNAP}/CollegeMsg.txt.gz', member=None),
    dict(name='LastFM-Asia', label='LastFM-Asia', url=f'{SNAP}/lastfm_asia.zip', member='lastfm_asia_edges.csv'),
    dict(name='Deezer-Europe', label='Deezer-Europe', url=f'{SNAP}/deezer_europe.zip', member='deezer_europe_edges.csv'),
    dict(name='Facebook-page-page', label='Facebook page--page', url=f'{SNAP}/facebook_large.zip', member='musae_facebook_edges.csv'),
    dict(name='Amazon0302', label='Amazon0302', url=f'{SNAP}/amazon0302.txt.gz', member=None),
    dict(name='Wiki-Talk', label='Wiki-Talk', url=f'{SNAP}/wiki-Talk.txt.gz', member=None),
    dict(name='com-Youtube', label='com-Youtube', url=f'{SNAP}/bigdata/communities/com-youtube.ungraph.txt.gz', member=None),
    dict(name='web-Google', label='web-Google', url=f'{SNAP}/web-Google.txt.gz', member=None),
    dict(name='roadNet-CA', label='roadNet-CA', url=f'{SNAP}/roadNet-CA.txt.gz', member=None),
]
MAX_EDGES_FOR_OPTIMIZATION = 20_000_000

SPLITS = [  # Part A: (split label, artifact manifest dir, artifact results dir)
    ('Development', 'development_benchmark', 'final_development'),
    ('Validation', 'real_benchmark', 'final_validation'),
    ('Audit', 'heldout_benchmark', 'final_heldout'),
]
# URLs used by the artifact's scripts/prepare_data.py for the development suite
FACEBOOK_URL = f'{SNAP}/facebook.tar.gz'
COLLABORATION = {'CA-GrQc.txt': f'{SNAP}/ca-GrQc.txt.gz', 'CA-CondMat.txt': f'{SNAP}/ca-CondMat.txt.gz',
                 'CA-HepPh.txt': f'{SNAP}/ca-HepPh.txt.gz'}


class GraphMissing(Exception):
    pass


def _test_override(url):
    """TESTING ONLY: PC_EVAL_TEST_BASEURL replaces the SNAP prefix (results are then flagged and make_tables.py
    refuses them for the paper)."""
    base = os.environ.get('PC_EVAL_TEST_BASEURL')
    if base and url.startswith(SNAP):
        return base.rstrip('/') + url[len(SNAP):], True
    return url, False


def _waits():
    return (0.05, 0.05, 0.05, 0.05) if os.environ.get('PC_EVAL_TEST_BASEURL') else (5, 10, 20, 40)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def download(url, dest, log, attempts=5, waits=None):
    """Download with retries through an atomic temporary file. Returns list of attempt records."""
    waits = waits or _waits()
    url, _ = _test_override(url)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i in range(attempts):
        tmp = None
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r, tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as t:
                tmp = Path(t.name)
                shutil.copyfileobj(r, t, 1 << 20)
            os.replace(tmp, dest)
            records.append(dict(attempt=i + 1, ok=True, time=time.time()))
            log(f'  downloaded {url} -> {dest} ({dest.stat().st_size:,} bytes)')
            return records
        except Exception as ex:  # noqa: BLE001
            if tmp is not None and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            records.append(dict(attempt=i + 1, ok=False, error=repr(ex), time=time.time()))
            log(f'  download attempt {i + 1}/{attempts} failed for {url}: {ex!r}')
            if i + 1 < attempts:
                time.sleep(waits[min(i, len(waits) - 1)])
    raise GraphMissing(f'download failed after {attempts} attempts: {url}')


def decompress_to(src, member, dest):
    """Write the edge file (gzip stream, or the unique zip member with basename `member`) to dest.
    Returns the decompressed size in bytes."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + '.writing')
    if member is None:
        with gzip.open(src, 'rb') as s, open(tmp, 'wb') as t:
            shutil.copyfileobj(s, t, 1 << 20)
    else:
        with zipfile.ZipFile(src) as z:
            hits = [i for i in z.infolist() if not i.is_dir() and Path(i.filename).name == member]
            if len(hits) != 1:
                raise GraphMissing(f'{src}: expected exactly one member named {member}, found {len(hits)}')
            with z.open(hits[0]) as s, open(tmp, 'wb') as t:
                shutil.copyfileobj(s, t, 1 << 20)
    os.replace(tmp, dest)
    return dest.stat().st_size


# ------------------------------------------------------------------------------------------------ parsing
def parse_edge_file(path):
    """Raw (u, v) arrays using the artifact's own parser dove.graph._edge_stream."""
    from dove.graph import _edge_stream
    flat = np.fromiter(itertools.chain.from_iterable(_edge_stream(Path(path))), dtype=np.int64)
    return flat[0::2].copy(), flat[1::2].copy()


def preprocess(us, vs):
    """Vectorised dove.graph.graph_from_edges: returns (dense edges int32 (m,2) sorted, original_ids int64)."""
    us = np.asarray(us, dtype=np.int64)
    vs = np.asarray(vs, dtype=np.int64)
    keep = us != vs
    a = np.minimum(us[keep], vs[keep])
    b = np.maximum(us[keep], vs[keep])
    if a.size == 0:
        raise ValueError('no usable undirected edges')
    ix = np.lexsort((b, a))
    a, b = a[ix], b[ix]
    first = np.ones(a.size, dtype=bool)
    first[1:] = (a[1:] != a[:-1]) | (b[1:] != b[:-1])
    a, b = a[first], b[first]
    ids = np.unique(np.concatenate([a, b]))
    da = np.searchsorted(ids, a)
    db = np.searchsorted(ids, b)
    if ids.size >= 2 ** 31:
        raise ValueError('too many vertices for int32 ids')
    return np.stack([da, db], axis=1).astype(np.int32), ids.astype(np.int64)


def make_graphdata(name, edges, original_ids, family='unknown'):
    """dove.graph.GraphData with the same adjacency layout (rows sorted ascending, int32)."""
    from dove.graph import GraphData
    n = int(original_ids.shape[0])
    e = np.asarray(edges, dtype=np.int64)
    src = np.concatenate([e[:, 0], e[:, 1]])
    dst = np.concatenate([e[:, 1], e[:, 0]])
    ix = np.lexsort((dst, src))
    nbr = dst[ix].astype(np.int32)
    counts = np.bincount(src, minlength=n)
    splits = np.cumsum(counts)[:-1]
    adjacency = tuple(np.split(nbr, splits))
    return GraphData(name, np.ascontiguousarray(edges, dtype=np.int32), adjacency, original_ids, family)


def load_graph_file(path, name, family='unknown'):
    us, vs = parse_edge_file(path)
    edges, ids = preprocess(us, vs)
    return edges, ids


def save_cache(path, edges, ids, meta):
    path = Path(path)
    tmp = path.with_name(path.stem + '.writing.npz')
    np.savez(tmp, edges=edges, original_ids=ids, meta=np.array(repr(meta)))
    os.replace(tmp, path)


def load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        return z['edges'], z['original_ids']


# ------------------------------------------------------------------------------------------------ Part A (artifact suite)
def read_manifest(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def read_per_graph(path):
    with open(path, newline='', encoding='utf-8') as f:
        return {r['graph']: r for r in csv.DictReader(f)}


def suite_plan(artifact):
    """List of dicts for the 32 suite graphs in manifest order."""
    art = Path(artifact)
    plan = []
    for split, mdir, rdir in SPLITS:
        rows = read_manifest(art / 'data' / mdir / 'manifest.csv')
        pg = read_per_graph(art / 'results' / rdir / 'per_graph.csv')
        for r in rows:
            name = r['name']
            item = dict(name=name, label=name, split=split, family=r.get('family', ''), manifest_path=r['path'],
                        results_dir=str(art / 'results' / rdir), expected_n=int(pg[name]['n']),
                        expected_m=int(pg[name]['m']), expected_neigh_best_work=int(pg[name]['neigh_best_work']),
                        expected_neigh_best_seed=int(pg[name]['neigh_best_seed']),
                        artifact_reference_work=int(pg[name]['reference_work']))
            if mdir == 'development_benchmark':
                fname = Path(r['path']).name
                if fname.endswith('.edges'):
                    item.update(kind='facebook', file=fname, url=FACEBOOK_URL)
                else:
                    item.update(kind='gz', file=fname, url=COLLABORATION[fname])
            else:
                item.update(kind='gz', file=Path(r['path']).name, url=r['source_url'],
                            expected_sha256=r['compressed_sha256'], expected_bytes=int(r['edge_file_bytes']))
            plan.append(item)
    return plan


def fetch_suite_graph(item, data_dir, log):
    """Download (if needed) and decompress one suite graph; returns (edge file path, provenance dict)."""
    raw = Path(data_dir) / 'partA' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    prov = dict(url=item['url'], test_url_override=_test_override(item['url'])[1])
    if item['kind'] == 'facebook':
        tar = raw / 'facebook.tar.gz'
        edir = raw / 'facebook'
        target = edir / item['file']
        if not tar.exists():
            prov['download'] = download(item['url'], tar, log)
        if not target.exists():
            edir.mkdir(exist_ok=True)
            with tarfile.open(tar, 'r:gz') as t:
                for mem in t.getmembers():
                    if mem.isfile() and Path(mem.name).suffix == '.edges':
                        src = t.extractfile(mem)
                        with src, open(edir / Path(mem.name).name, 'wb') as out:
                            shutil.copyfileobj(src, out)
        if not target.exists():
            raise GraphMissing(f'{item["file"]} not found in facebook.tar.gz')
        prov['archive_sha256'] = sha256_file(tar)
        prov['edge_file_bytes'] = target.stat().st_size
        return target, prov
    comp = raw / Path(item['url']).name
    target = raw / item['file']
    if not comp.exists():
        prov['download'] = download(item['url'], comp, log)
    prov['compressed_sha256'] = sha256_file(comp)
    if not target.exists():
        decompress_to(comp, None, target)
    prov['edge_file_bytes'] = target.stat().st_size
    if 'expected_sha256' in item:
        prov['sha256_matches_manifest'] = prov['compressed_sha256'] == item['expected_sha256']
        prov['bytes_match_manifest'] = prov['edge_file_bytes'] == item['expected_bytes']
        if not prov['sha256_matches_manifest']:
            log(f'  WARNING {item["name"]}: compressed SHA-256 differs from the artifact manifest')
        if not prov['bytes_match_manifest']:
            log(f'  WARNING {item["name"]}: decompressed size differs from the artifact manifest')
    return target, prov


def fetch_heldout_graph(item, data_dir, log):
    raw = Path(data_dir) / 'partB' / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    comp = raw / Path(item['url']).name
    prov = dict(url=item['url'], member=item['member'], test_url_override=_test_override(item['url'])[1])
    if not comp.exists():
        prov['download'] = download(item['url'], comp, log)
    prov['downloaded_sha256'] = sha256_file(comp)
    prov['downloaded_bytes'] = comp.stat().st_size
    target = raw / (item['member'] if item['member'] else Path(item['url']).name[:-3])
    try:
        prov['edge_file_bytes'] = decompress_to(comp, item['member'], target)
    except GraphMissing:
        raise
    except Exception as ex:  # noqa: BLE001
        raise GraphMissing(f'decompression failed: {ex!r}')
    prov['edge_file_sha256'] = sha256_file(target)
    return target, prov
