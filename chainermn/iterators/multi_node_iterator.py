import enum

import chainer
import numpy


class _DataType(enum.IntEnum):
    FLOAT32_FLOAT32 = 1
    FLOAT32_INT32 = 2
    FLOAT32 = 3
    OTHER = 4
    INVALID = 5


def _get_data_type(element):
    if isinstance(element, tuple) and len(element) == 2 \
            and hasattr(element[0], 'dtype') \
            and element[0].dtype == numpy.float32 \
            and hasattr(element[1], 'dtype') \
            and element[1].dtype == numpy.float32:
        return _DataType.FLOAT32_FLOAT32
    elif isinstance(element, tuple) and len(element) == 2 \
            and hasattr(element[0], 'dtype') \
            and element[0].dtype == numpy.float32 \
            and hasattr(element[1], 'dtype') \
            and element[1].dtype == numpy.int32:
        return _DataType.FLOAT32_INT32
    elif hasattr(element, 'dtype') and element.dtype == numpy.float32:
        return _DataType.FLOAT32
    else:
        return _DataType.OTHER


def _build_ctrl_msg(stop, data_type, is_new_epoch, current_position):
    ctrl_msg = numpy.ones((4,)) \
        * [int(stop), int(data_type), int(is_new_epoch), int(current_position)]
    ctrl_msg = ctrl_msg.astype(numpy.int32)
    return ctrl_msg.view(numpy.float32)


def _parse_ctrl_msg(msg):
    msg = msg.view(numpy.int32)
    stop = bool(msg[0])
    data_type = _DataType(msg[1])
    is_new_epoch = bool(msg[2])
    current_position = int(msg[3])
    return stop, data_type, is_new_epoch, current_position


class _MultiNodeIteratorMaster(chainer.dataset.iterator.Iterator):

    def __init__(self, actual_iterator, communicator, rank_master):
        super(_MultiNodeIteratorMaster, self).__setattr__(
            'communicator', communicator)
        super(_MultiNodeIteratorMaster, self).__setattr__(
            'actual_iterator', actual_iterator)
        super(_MultiNodeIteratorMaster, self).__setattr__(
            'rank_master', rank_master)

        _dataset_size = numpy.ones((1, )).astype(numpy.float32) \
            * len(self.actual_iterator.dataset)
        # TODO(tsutsumi): potential deadlock?
        self.communicator.bcast(_dataset_size, root=self.rank_master)
        if self.actual_iterator._order is not None:
            self.communicator.bcast(
                self.actual_iterator._order.astype(numpy.float32),
                root=self.rank_master)
        else:
            # Without shuffle, order is None.
            self.communicator.bcast(
                -numpy.ones((1, )).astype(numpy.float32),
                root=self.rank_master)

    def __next__(self):
        try:
            batch = self.actual_iterator.__next__()
            first_elem = batch[0]
            data_type = _get_data_type(first_elem)
            stop = False
        except StopIteration:
            data_type = _DataType.INVALID
            stop = True

        is_new_epoch = self.actual_iterator.is_new_epoch
        ctrl_msg = _build_ctrl_msg(stop, data_type, is_new_epoch,
                                   self.actual_iterator.current_position)
        self.communicator.bcast(ctrl_msg, root=self.rank_master)

        if stop:
            raise StopIteration

        if data_type == _DataType.FLOAT32_FLOAT32:
            _xs, _ys = zip(*batch)
            xs = numpy.asarray(_xs, dtype=numpy.float32)
            ys = numpy.asarray(_ys, dtype=numpy.float32)
            self.communicator.bcast(xs, root=self.rank_master)
            self.communicator.bcast(ys, root=self.rank_master)
            return batch
        elif data_type == _DataType.FLOAT32_INT32:
            _xs, _ys = zip(*batch)
            xs = numpy.asarray(_xs, dtype=numpy.float32)
            ys = numpy.asarray(_ys, dtype=numpy.int32).view(numpy.float32)
            self.communicator.bcast(xs, root=self.rank_master)
            self.communicator.bcast(ys, root=self.rank_master)
            return batch
        elif data_type == _DataType.FLOAT32:
            if isinstance(batch, list):
                batch = numpy.array(batch)
            batch = self.communicator.bcast(batch, root=self.rank_master)
            return batch.tolist()
        elif data_type == _DataType.OTHER:
            batch = self.communicator.bcast_obj(batch, root=self.rank_master)
            return batch

    next = __next__

    def __getattr__(self, attr_name):
        return getattr(self.actual_iterator, attr_name)

    def __setattr_(self, attr_name, value):
        setattr(self.actual_iterator, attr_name, value)

    @property
    def current_position(self):
        return self.actual_iterator.current_position

    @property
    def epoch_detail(self):
        return self.actual_iterator.epoch_detail

    @property
    def is_new_epoch(self):
        return self.actual_iterator.is_new_epoch

    def serialize(self, serializer):
        # Master's and Slave's serialize must be called at the same time.
        self.actual_iterator.serialize(serializer)
        self.communicator.bcast_obj(
            serializer, root=self.rank_master)


class _MultiNodeIteratorSlave(chainer.dataset.iterator.Iterator):

    def __init__(self, communicator, rank_master):
        super(_MultiNodeIteratorSlave, self).__init__()
        self.communicator = communicator
        self.rank_master = rank_master

        # Compatibility to Chainer iterators.
        self.epoch = 0
        self.current_position = 0
        self.is_new_epoch = False

        # TODO(tsutsumi): potential deadlock?
        _size = self.communicator.bcast(None, root=self.rank_master)
        self.dataset_size = int(_size)
        self._order = self.communicator.bcast(None, root=self.rank_master)
        self._order = self._order.astype(numpy.int64)
        if self._order[0] == -1:
            self._order = None

    def __next__(self):
        # Check if master iterator received stop signal.
        ctrl_msg = self.communicator.bcast(None, root=self.rank_master)
        stop, data_type, self.is_new_epoch, \
            self.current_position = _parse_ctrl_msg(ctrl_msg)

        if self.is_new_epoch:
            self.epoch += 1

        if stop:
            raise StopIteration

        if data_type == _DataType.FLOAT32_FLOAT32:
            xs = self.communicator.bcast(None, root=self.rank_master)
            ys = self.communicator.bcast(None, root=self.rank_master)
            return list(zip(xs, ys.astype(numpy.int32)))
        elif data_type == _DataType.FLOAT32_INT32:
            xs = self.communicator.bcast(None, root=self.rank_master)
            ys = self.communicator.bcast(None, root=self.rank_master)
            return list(zip(xs, ys.view(numpy.int32)))
        elif data_type == _DataType.FLOAT32:
            batch = self.communicator.bcast(None, root=self.rank_master)
            return batch.tolist()
        elif data_type == _DataType.OTHER:
            batch = self.communicator.bcast_obj(None, root=self.rank_master)
            return batch

    @property
    def epoch_detail(self):
        return self.epoch + 1. * self.current_position / self.dataset_size

    def serialize(self, serializer):
        # Master's and Slave's serialize must be called at the same time.
        _serializer = self.communicator.bcast_obj(
            None, root=self.rank_master)

        self.current_position = serializer(
            'current_position',
            _serializer('current_position', self.current_position)
        )
        self.epoch = serializer('epoch', _serializer('epoch', self.epoch))
        self.is_new_epoch = serializer(
            'is_new_epoch',
            _serializer('is_new_epoch', self.is_new_epoch)
        )

        try:
            self._order = serializer(
                'order',
                _serializer('order', self._order)
            )
        except KeyError:
            pass


def create_multi_node_iterator(
        actual_iterator, communicator, rank_master=0):
    """Create a multi node iterator from a Chainer iterator.

    This iterator shares the same batches on multiple processes, simply
    broadcasting batches from master process to slave processes
    in each iteration.
    Master process obtains batches from ``actual_iterator``, which you can
    specify any Chainer iterator (e.g. ``chainer.iterators.SerialIterator``).

    Here is an example situation. When we train a sequence-to-sequence model,
    where the encoder and the decoder is located on two different processes,
    we want to share the same batches on each process, thus inputs for
    the encoder and output teacher signals for the decoder become consistent.

    In order to use the multi node iterator, first create the iterator
    from Chainer iterator and ChainerMN communicator::

        iterator = chainermn.iterators.create_multi_node_iterator(
            chainer.iterators.SerialIterator(
                dataset, batch_size, shuffle=True),
            communicator)

    Then you can use it as the ordinary Chainer iterator::

        updater = chainer.training.StandardUpdater(iterator, optimizer)
        trainer = training.Trainer(updater)
        trainer.run()

    Since this iterator shares batches through network in each iteration,
    communication might be large. If you train your model-parallel network
    on extremely large dataset, you can also consider to use
    ``chainermn.iterators.create_synchronized_iterator``.

    Current multi node iterator supports numpy.float32 or tuple of
    numpy.float32 as the data type of the batch element.

    .. note:: ``create_multi_node_iterator`` and ``serialize`` of created
              iterators must be called at the same time by master and slaves,
              unless it falls into deadlock because they synchronize internal
              states of iterators.

    Args:
        actual_iterator: Chainer iterator
            (``chainer.iterators.SerialIterator`` and
            ``chainer.iterators.MultiprocessIterator`` are supported).
        communicator: ChainerMN communicator.
        rank_master: process rank to be master.

    Returns:
        The master-slave iterator based on ``actual_iterator``.
    """
    chainer.utils.experimental(
        'chainermn.iterators.create_multi_node_iterator')
    if communicator.rank == rank_master:
        return _MultiNodeIteratorMaster(
            actual_iterator, communicator, rank_master)
    else:
        return _MultiNodeIteratorSlave(communicator, rank_master)
