'''
Aggregate per-point result.json files into a single table and report scan status.
Replaces the old gather_reports() and the bespoke rescan failure-analysis script.
'''
import json
import logging
import os

from conifer.utils.performance.scan.manifest import DEFAULT_RESULT_NAME, Manifest, points_dir

logger = logging.getLogger(__name__)


def iter_results(scandir):
  '''Yield every parseable result.json under the scan's points directory.'''
  pdir = points_dir(scandir)
  if not os.path.isdir(pdir):
    return
  for pid in sorted(os.listdir(pdir)):
    rp = os.path.join(pdir, pid, DEFAULT_RESULT_NAME)
    if os.path.isfile(rp):
      try:
        yield json.load(open(rp))
      except json.JSONDecodeError:
        logger.warning(f'unparseable result {rp}')


def gather(scandir, out=None, ok_only=False):
  '''Collect results into a DataFrame and write results.csv (+ results.parquet if pyarrow present).'''
  import pandas as pd
  rows = [r for r in iter_results(scandir) if not ok_only or r.get('outcome') == 'ok']
  if not rows:
    logger.warning(f'no results found under {scandir}')
  df = pd.json_normalize(rows)
  # ModelMetrics/SynthesisMetrics.schema_version are the source of truth; mixed versions need attention
  for col in ('model_metrics_schema_version', 'synthesis_metrics_schema_version'):
    n_schemas = df[col].nunique() if col in df else 0
    if n_schemas > 1:
      logger.warning(f'{n_schemas} different {col} values in this scan; '
                     'points were built from code that changed feature/label fields mid-scan')
  out = out or os.path.join(scandir, 'results')
  df.to_csv(f'{out}.csv', index=False)
  try:
    df.to_parquet(f'{out}.parquet', index=False)
  except Exception as e:
    logger.warning(f'parquet not written ({e}); install conifer[scan] for pyarrow')
  print(f'gathered {len(df)} results -> {out}.csv')
  return df


def status_report(scandir):
  '''Per-point outcome for the whole manifest, including points that never started.'''
  import pandas as pd
  manifest = Manifest.load(scandir)
  done = {r['point_id']: r for r in iter_results(scandir)}
  rows = []
  for p in manifest:
    r = done.get(p.point_id)
    rows.append({'point_id': p.point_id, 'name': p.name, 'trial': p.trial,
                 'outcome': r['outcome'] if r else 'not_started',
                 'reason': (r or {}).get('reason', ''),
                 'wall_time_s': (r or {}).get('wall_time_s')})
  df = pd.DataFrame(rows)

  total = len(df)
  counts = df['outcome'].value_counts()
  print(f'{scandir}: {total} points')
  for outcome, n in counts.items():
    print(f'  {outcome:<12} {n:>6}  ({100 * n / total:5.1f}%)')
  reasons = df.loc[~df['outcome'].isin(('ok', 'not_started')) & (df['reason'] != ''), 'reason']
  if len(reasons):
    print('failure reasons:')
    for reason, n in reasons.value_counts().items():
      print(f'  {n:>6}  {reason}')
  return df


def add_gather_arguments(parser):
  parser.add_argument('scandir')
  parser.add_argument('-o', '--out', help='output path stem (default SCANDIR/results)')
  parser.add_argument('--ok-only', action='store_true', help='keep only successful points')


def gather_from_args(args):
  return gather(args.scandir, out=args.out, ok_only=args.ok_only)
