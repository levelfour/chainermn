import chainer
import chainer.testing
import chainermn
import numpy as np
import pytest
import unittest


class TestSynchronizedIterator(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip("This test is for multinode only")

        N = 100
        self.dataset = np.arange(N).astype(np.float32)

    def test_sync_no_repeat(self):
        iterator = chainermn.iterators.create_synchronized_iterator(
            chainer.iterators.SerialIterator(self.dataset, batch_size=4, shuffle=True, repeat=False),
            self.communicator)

        for e in range(3):
            try:
                while True:
                    batch = np.array(iterator.next(), dtype=np.float32)
                    if self.communicator.rank == 0:
                        for rank_from in range(1, self.communicator.size):
                            _batch = self.communicator.recv(source=rank_from, tag=0)
                            chainer.testing.assert_allclose(batch, _batch)
                    else:
                        self.communicator.send(batch, dest=0, tag=0)
            except StopIteration:
                iterator.reset()

    def test_sync_no_repeat_frag(self):
        iterator = chainermn.iterators.create_synchronized_iterator(
            chainer.iterators.SerialIterator(self.dataset, batch_size=7, shuffle=True, repeat=False),
            self.communicator)

        for e in range(3):
            try:
                while True:
                    batch = np.array(iterator.next(), dtype=np.float32)
                    if self.communicator.rank == 0:
                        for rank_from in range(1, self.communicator.size):
                            _batch = self.communicator.recv(source=rank_from, tag=0)
                            chainer.testing.assert_allclose(batch, _batch)
                    else:
                        self.communicator.send(batch, dest=0, tag=0)
            except StopIteration:
                iterator.reset()
