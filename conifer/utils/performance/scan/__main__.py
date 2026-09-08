'''conifer performance-scan CLI: expand a spec, plan/shard it, run it, gather results.'''
import argparse
import logging

from conifer.utils.performance.scan.gather import add_gather_arguments, gather_from_args, status_report
from conifer.utils.performance.scan.manifest import expand_spec
from conifer.utils.performance.scan.plan import add_plan_arguments, plan_from_args
from conifer.utils.performance.scan.run import add_arguments as add_run_arguments
from conifer.utils.performance.scan.run import run_from_args


def main(argv=None):
  logging.basicConfig(level=logging.INFO, format='%(message)s')
  parser = argparse.ArgumentParser(prog='python -m conifer.utils.performance.scan', description=__doc__)
  sub = parser.add_subparsers(dest='cmd', required=True)

  p = sub.add_parser('expand', help='expand a ScanSpec into scandir/manifest.jsonl')
  p.add_argument('spec')
  p.add_argument('scandir')
  p.set_defaults(func=lambda a: expand_spec(a.spec, a.scandir))

  p = sub.add_parser('plan', help='cost estimate and optional LPT sharding')
  add_plan_arguments(p)
  p.set_defaults(func=plan_from_args)

  p = sub.add_parser('run', help='build pending points of a scan or shard')
  add_run_arguments(p)
  p.set_defaults(func=run_from_args)

  p = sub.add_parser('gather', help='aggregate result.json files into results.csv/.parquet')
  add_gather_arguments(p)
  p.set_defaults(func=gather_from_args)

  p = sub.add_parser('status', help='progress + per-outcome breakdown for the whole manifest')
  p.add_argument('scandir')
  p.add_argument('--watch', type=int, metavar='SECONDS', help='redraw every N seconds until ctrl-c')
  p.set_defaults(func=lambda a: status_report(a.scandir, watch=a.watch))

  args = parser.parse_args(argv)
  args.func(args)


if __name__ == '__main__':
  main()
