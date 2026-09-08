'''
Run a scan manifest, or a round-robin shard of it, with a pool of workers.
Resumable: points that already have a result.json are skipped.
'''
import concurrent.futures
import json
import os

from conifer.utils.performance.scan.build import run_point
from conifer.utils.performance.scan.manifest import DEFAULT_SPEC_NAME, Manifest

_OUTCOMES = ('ok', 'hls_fail', 'vsynth_fail', 'timeout', 'oom', 'error')


def load_base_config(scandir, override=None):
  '''Return the conifer base config for a scan: spec.yml's base_config, optionally overridden by a file.'''
  cfg = {}
  spec_path = os.path.join(scandir, DEFAULT_SPEC_NAME)
  if os.path.isfile(spec_path):
    import yaml
    cfg = (yaml.safe_load(open(spec_path)) or {}).get('base_config', {}) or {}
  if override:
    if str(override).endswith(('.yml', '.yaml')):
      import yaml
      cfg = {**cfg, **(yaml.safe_load(open(override)) or {})}
    else:
      cfg = {**cfg, **json.load(open(override))}
  return cfg


def run_manifest(scandir, shard=None, shard_file=None, jobs=1, timeout=None, mem_gb=None,
                 do_vsynth=True, resume=True, isolate=True, config=None, progress=True):
  '''Build every pending point of the scan (or shard) and return a {outcome: count} summary.'''
  manifest = Manifest.load(shard_file) if shard_file else Manifest.load(scandir)
  base_config = load_base_config(scandir, config)
  if shard is not None and not shard_file:
    manifest = manifest.shard(shard[0], shard[1])
  todo = manifest.pending(scandir) if resume else manifest
  n_total, n_skipped = len(manifest), len(manifest) - len(todo)

  bar = None
  if progress:
    try:
      from tqdm import tqdm
      bar = tqdm(total=len(todo), desc=f'\N{evergreen tree} scan {os.path.basename(scandir.rstrip("/"))}')
    except ImportError:
      pass

  counts = {k: 0 for k in _OUTCOMES}
  with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
    futs = {ex.submit(run_point, p, scandir, base_config, timeout, mem_gb, do_vsynth, None, isolate): p
            for p in todo}
    for fut in concurrent.futures.as_completed(futs):
      res = fut.result()
      counts[res.get('outcome', 'error')] = counts.get(res.get('outcome', 'error'), 0) + 1
      if bar is not None:
        bar.update(1)
        bar.set_postfix({k: v for k, v in counts.items() if v})
  if bar is not None:
    bar.close()

  summary = {'scandir': scandir, 'points': n_total, 'skipped': n_skipped, 'run': len(todo), **counts}
  print(f'{summary}')
  return summary


def _parse_shard(s):
  '''Parse an "i/N" shard string into a 0-based (index, n_shards) tuple.'''
  i, n = s.split('/')
  return int(i), int(n)


def add_arguments(parser):
  parser.add_argument('scandir', help='scan directory containing manifest.jsonl')
  parser.add_argument('--shard', type=_parse_shard, help='round-robin shard as i/N (0-based i)')
  parser.add_argument('--shard-file', help='run the points listed in this jsonl (e.g. from plan)')
  parser.add_argument('-j', '--jobs', type=int, default=1, help='concurrent points')
  parser.add_argument('--timeout', type=float, help='per-point wall-clock limit (s)')
  parser.add_argument('--mem-gb', type=float, help='per-point address-space cap (GB)')
  parser.add_argument('--no-vsynth', action='store_true', help='skip Vivado synthesis (HLS only)')
  parser.add_argument('--no-resume', action='store_true', help='rebuild points that already have a result')
  parser.add_argument('--no-isolate', action='store_true', help='build in-process (no timeout/mem cap)')
  parser.add_argument('-c', '--config', help='conifer config file overriding the spec base_config')


def run_from_args(args):
  return run_manifest(args.scandir, shard=args.shard, shard_file=args.shard_file, jobs=args.jobs,
                      timeout=args.timeout, mem_gb=args.mem_gb, do_vsynth=not args.no_vsynth,
                      resume=not args.no_resume, isolate=not args.no_isolate, config=args.config)


if __name__ == '__main__':
  import argparse
  p = argparse.ArgumentParser(description=__doc__)
  add_arguments(p)
  run_from_args(p.parse_args())
