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


def _bar(frac, width=34):
  n = int(round(max(0.0, min(1.0, frac)) * width))
  return '[' + '#' * n + '-' * (width - n) + ']'


def _render_status(scandir):
  '''Print a one-shot status view and return the per-point DataFrame.'''
  import pandas as pd
  manifest = Manifest.load(scandir)
  done = {r['point_id']: r for r in iter_results(scandir)}
  rows = []
  for p in manifest:
    r = done.get(p.point_id)
    # 'pending' = no result.json yet: either not started or being synthesized right now
    # (a running point leaves no marker on EOS, so the two can't be told apart from here)
    rows.append({'point_id': p.point_id, 'outcome': r['outcome'] if r else 'pending',
                 'reason': (r or {}).get('reason', ''),
                 'finished_at': (r or {}).get('finished_at'),
                 'wall_time_s': (r or {}).get('wall_time_s')})
  df = pd.DataFrame(rows)
  total = len(df)
  n_done = int((df['outcome'] != 'pending').sum()) if total else 0
  frac = n_done / total if total else 0.0

  print(f'{os.path.basename(scandir.rstrip("/"))}: {n_done}/{total} ({100 * frac:.1f}%)  {_bar(frac)}')
  for outcome, n in df['outcome'].value_counts().items():
    label = 'pending (or running)' if outcome == 'pending' else outcome
    print(f'  {label:<20} {n:>6}  ({100 * n / total:5.1f}%)')

  fin = pd.to_datetime(df['finished_at'], errors='coerce', utc=True).dropna()
  if len(fin) and n_done < total:
    recent = int((fin > pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=10)).sum())
    rate = recent / 10.0
    eta = f'~{(total - n_done) / rate:.0f} min' if rate > 0 else 'n/a'
    note = '' if rate > 0 else '  <- nothing finished recently; check the runners'
    print(f'  last 10 min : {recent:>6}  ({rate:.1f}/min, ETA {eta}){note}')

  reasons = df.loc[~df['outcome'].isin(('ok', 'pending')) & (df['reason'] != ''), 'reason']
  if len(reasons):
    print('failure reasons:')
    for reason, n in reasons.value_counts().head(10).items():
      print(f'  {n:>6}  {reason}')
  return df


def status_report(scandir, watch=None):
  '''Per-point outcome for the whole manifest; with watch=<seconds>, redraw until ctrl-c.'''
  if not watch:
    return _render_status(scandir)
  import time
  import datetime
  df = None
  try:
    while True:
      print(f'\033[2J\033[H(watch {watch}s, ctrl-c to stop)  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')
      df = _render_status(scandir)
      time.sleep(watch)
  except KeyboardInterrupt:
    return df if df is not None else _render_status(scandir)


def add_gather_arguments(parser):
  parser.add_argument('scandir')
  parser.add_argument('-o', '--out', help='output path stem (default SCANDIR/results)')
  parser.add_argument('--ok-only', action='store_true', help='keep only successful points')


def gather_from_args(args):
  return gather(args.scandir, out=args.out, ok_only=args.ok_only)
