"""Tests for torch_xla.distributed.sharded."""
import logging
import os
import unittest

import torch
import torch_xla
import torch_xla.core.xla_model as xm


class SpmdTest(unittest.TestCase):

  def setUp(self):
    super().setUp()

  def teardown(self):
    super().teardown()

  def test_sharded_add(self):
    t = torch.arange(64, dtype=torch.float32).reshape((8,8))
    pass

  def test_convolution(self):
    # Goal: determine where to call _XLAC._xla_set_sharding() from
    # (before or after tracing IR?)
    # A: Lowering of nodes happens only at LoweringContext() creation time
    # which seems to be around compilation time (double check) so as long
    # as its set before compilation we *should* be good.
    torch_xla._XLAC.sharding.set_sharding([2,1]);

    dev = xm.xla_device()

    x = torch.rand((1,3,10,10)).to(dev)
    conv = torch.nn.Conv2d(3, 5, kernel_size=3).to(dev)

    y = conv(x)
    # print(torch_xla._XLAC._get_xla_tensors_hlo([y]))

    xm.mark_step()


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.INFO)
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)
