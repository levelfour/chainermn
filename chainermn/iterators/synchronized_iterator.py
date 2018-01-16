import numpy


class _SynchronizedIterator(object):

    def __init__(self, actual_iterator, communicator):
        if not hasattr(actual_iterator, 'set_random_state'):
            raise ValueError('actual_iterator must have set_random_state()')
        else:
            super(_SynchronizedIterator, self).__setattr__(
                'actual_iterator', actual_iterator)

        # Synchronize random seed.
        self.communicator = communicator
        if self.communicator.rank == 0:
            seed = numpy.random.randint(0, 2 ** 32 - 1)
        else:
            seed = None
        seed = self.communicator.mpi_comm.bcast(seed, root=0)

        # Random number generator for iterator.
        rng = numpy.random.RandomState(seed)
        self.actual_iterator.set_random_state(rng)

    def __getattr__(self, attr_name):
        return getattr(self.actual_iterator, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.actual_iterator, attr_name, value)


def create_synchronized_iterator(actual_iterator, communicator):
    """Create a synchronized iterator from a Chainer iterator.
    This is used when you want batches on multiple processes
    to be synchronized.

    Args:
        actual_iterator: Chainer iterator
            (e.g., ``chainer.iterators.SerialIterator``).
        communicator: ChainerMN communicator.

    Returns:
        The synchronized iterator based on ``actual_iterator``.
    """
    return _SynchronizedIterator(actual_iterator, communicator)
