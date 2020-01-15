#!/usr/bin/env python
"""User tool to distribute a command among all instances in instance group."""

import argparse
from torch_xla.distributed.xla_dist import Cluster
from torch_xla.distributed.xla_dist import ClusterResolver
from torch_xla.distributed.xla_dist import DistributedExecutor


def distribute_cmds(args):
  cr = ClusterResolver('fake-tpu', zone=args.zone, project=args.project)
  cws = cr.get_client_workers()
  cluster = Cluster(cws, [])
  executor = DistributedExecutor(cluster)
  executor.run(args.cmdline)


if __name__ == '__main__':
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument(
    '--project',
    type=str,
    help='(Optional) Name of project that has the instance group.')
  arg_parser.add_argument(
    '--zone',
    type=str,
    help='(Optional) Zone that instance group belongs to (we use single zone).')
  arg_parser.add_argument('cmdline', nargs='+')

  args = arg_parser.parse_args()
  distribute_cmds(args)
