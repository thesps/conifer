'''Child-process entrypoint: build one scan point from a job JSON file written by run_point().'''
import json
import sys

from conifer.utils.performance.scan.build import build_point
from conifer.utils.performance.scan.manifest import Point


def main(jobfile):
  job = json.load(open(jobfile))
  point = Point.from_dict(job['point'])
  keep = set(job['keep']) if job.get('keep') else None
  build_point(point, job['root'], job['base_config'], do_vsynth=job.get('do_vsynth', True), keep=keep)


if __name__ == '__main__':
  main(sys.argv[1])
