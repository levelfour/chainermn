import collections

import mpi4py
import numpy

import chainer.cuda
import chainer.utils
from chainermn.communicators import _communication_utility
from chainermn.communicators._communication_utility import chunked_bcast_obj
from chainermn.communicators import _memory_utility
from chainermn.communicators import communicator_base


_dtype_mpi_type = {
    numpy.dtype(numpy.int8): mpi4py.MPI.INT8_T,
    numpy.dtype(numpy.int16): mpi4py.MPI.INT16_T,
    numpy.dtype(numpy.int32): mpi4py.MPI.INT32_T,
    numpy.dtype(numpy.int64): mpi4py.MPI.INT64_T,
    numpy.dtype(numpy.float32): mpi4py.MPI.FLOAT,
    numpy.dtype(numpy.float64): mpi4py.MPI.DOUBLE,
}


def _check_dtype(method_name, msgtype):
    dtype = msgtype.dtype
    if dtype not in _dtype_mpi_type.keys():
        raise TypeError(
            '{} does not support dtype {}'.format(method_name, dtype))


def _check_dtypes_are_same(msgtypes):
    dtypes = [msgtype.dtype for msgtype in msgtypes]
    if any(dtypes[0] != dtype for dtype in dtypes):
        raise TypeError('all dtypes must be the same')


def _cnt_to_dsp(cnt):
    """Utility to convert length array to cumulative array."""
    return [0] + numpy.cumsum(cnt)[:-1].tolist()


def _get_mpi_type(msgtype):
    dtype = msgtype.dtype
    if dtype not in _dtype_mpi_type.keys():
        raise TypeError(
            'dtype {} is not supported by MpiCommunicator'.format(dtype))

    return _dtype_mpi_type[dtype]


class _MessageType(object):

    def __init__(self, obj):
        if hasattr(obj, 'dtype'):
            self.is_tuple = False
            self.narr = 1
            self.ndims = [obj.ndim]
            self.shapes = [obj.shape]
            self.dtype = obj.dtype
        elif isinstance(obj, collections.Iterable) \
                and all(hasattr(x, 'dtype') for x in obj):
            self.is_tuple = True
            self.narr = len(obj)
            self.ndims = [x.ndim for x in obj]
            self.shapes = [x.shape for x in obj]
            dtypes = [x.dtype for x in obj]
            if not all(dtype == dtypes[0] for dtype in dtypes):
                raise TypeError(
                    'Message objects must be the same dtype')
            self.dtype = dtypes[0]
        else:
            raise TypeError(
                'Message object must be numpy/cupy array or its tuple.')


class MpiCommunicatorBase(communicator_base.CommunicatorBase):
    '''MpiCommunicatorBase

    Implementation of communicator interface defined by
    :class:`CommunicatorBase`. This communicator assumes MPI4py and
    all ChainerMN processes are invoked by ``mpirun`` (``mpiexec``)
    command. Although this lacks several important methods such as
    ``allreduce_grad`` to be impelmented with speficic algorithm. See
    hierarcical communicator or pure_nccl communicator for example.

    '''

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm
        self._init_ranks()

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def intra_rank(self):
        return self._intra_rank

    @property
    def size(self):
        return self.mpi_comm.size

    def split(self, color, key):
        return self.__class__(mpi_comm=self.mpi_comm.Split(color, key))

    def alltoall(self, xs):
        """A primitive of inter-process all-to-all function.

        This method tries to invoke all-to-all communication within the
        communicator. All processes in the communicator are expected to
        invoke ``alltoall()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            xs (tuple of numpy.ndarray)

        Returns:
            ys (tuple of numpy.ndarray):
                Received arrays. The length of tuple equals to
                the communicator size.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.alltoall')

        if len(xs) != self.size:
            raise ValueError(
                'The length of data must be same as communicator size.')

        # Type check.
        msgtypes = [_MessageType(x) for x in xs]
        for msgtype in msgtypes:
            _check_dtype('alltoall', msgtype)
        _check_dtypes_are_same(msgtypes)
        send_msgtype = msgtypes[0]

        msgtypes = self.mpi_comm.alltoall(msgtypes)
        _check_dtypes_are_same(msgtypes)
        recv_msgtype = msgtypes[0]

        # Collective communication.
        slens = [numpy.prod(x.shape) for x in xs]
        xp = chainer.cuda.get_array_module(*xs)
        sbuf = xp.hstack([x.reshape(-1) for x in xs])
        shapes = [msgtype.shapes[0] for msgtype in msgtypes]
        rlens = [numpy.prod(s) for s in shapes]
        rbuf = numpy.empty([sum(rlens)], dtype=msgtype.dtype)
        if xp is not numpy:
            sbuf = _memory_utility.array_to_buffer_object(sbuf)[0]
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Alltoallv(
            [sbuf, (slens, _cnt_to_dsp(slens)), _get_mpi_type(send_msgtype)],
            [rbuf, (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(recv_msgtype)])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    def send(self, data, dest, tag):
        """A primitive for inter-process transmitter.

        This method sends numpy-array to target process.
        The target process is expected to invoke ``recv()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            data: data to be sent (tuple, list or raw numpy/cupy array)
            dest (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.send')

        msgtype = _MessageType(data)
        _check_dtype('send', msgtype)

        """We use ssend() instead of send() to pass unittests.
        If we don't use it, an error occurs in
        test_point_to_point_communication.py
        when using MVAPICH2-2.2 and GPUs.
        """
        self.mpi_comm.ssend(msgtype, dest=dest, tag=tag)

        # Type check.
        if not msgtype.is_tuple:
            data = [data]

        for array in data:
            if chainer.cuda.get_array_module(array) is not numpy:
                chainer.cuda.Stream.null.synchronize()

            buf = _memory_utility.array_to_buffer_object(array)
            """We use Ssend() for the same reason as using ssend()."""
            self.mpi_comm.Ssend(buf, dest=dest, tag=tag)

    def recv(self, source, tag):
        """A primitive of inter-process receiver.

        This method tries to receive numpy-array from target process.
        The target process is expected to invoke ``send()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            source (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """

        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.recv')

        msgtype = self.mpi_comm.recv(source=source, tag=tag)

        if msgtype.is_tuple:
            msg = []
            for shape in msgtype.shapes:
                buf = numpy.empty(numpy.prod(shape), dtype=msgtype.dtype)
                self.mpi_comm.Recv(buf, source=source, tag=tag)
                msg.append(buf.reshape(shape))
            return tuple(msg)

        else:
            assert len(msgtype.shapes) == 1
            shape = msgtype.shapes[0]
            buf = numpy.empty(numpy.prod(shape), dtype=msgtype.dtype)
            self.mpi_comm.Recv(buf, source=source, tag=tag)
            return buf.reshape(shape)

    def bcast(self, x, root=0):
        """A primitive of inter-process broadcast communication.

        This method tries to invoke broadcast communication within the
        communicator. All processes in the communicator are expected to
        invoke ``broadcast()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            x (numpy.array): Array to be broadcasted.
            root (int): Rank of root process.

        Returns:
            ys (tuple of numpy.ndarray): Received arrays.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.bcast')

        is_master = self.mpi_comm.rank == root

        if is_master:
            msgtype = _MessageType(x)
            _check_dtype('bcast', msgtype)

            if msgtype.is_tuple:
                raise TypeError('Tuple data cannot be broadcasted')

            msgtype = self.mpi_comm.bcast(msgtype, root)
            shape = msgtype.shapes[0]
            buf = _memory_utility.array_to_buffer_object(x)
            self.mpi_comm.Bcast(buf, root)
            return x
        else:
            msgtype = self.mpi_comm.bcast(None, root)
            shape = msgtype.shapes[0]
            buf = numpy.empty(numpy.prod(shape), dtype=msgtype.dtype)
            self.mpi_comm.Bcast(buf, root)
            return buf.reshape(shape)

    def gather(self, x, root=0):
        """A primitive of inter-process gather communication.

        This method tries to invoke gather communication within the
        communicator. All processes in the communicator are expected to
        invoke ``gather()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Args:
            x (numpy.array): Array to be gathered.
            root (int): Rank of root process.

        Returns:
            ys (tuple of numpy.ndarray):
                Received arrays. ``None`` for non-root processes.
        """
        chainer.utils.experimental(
            'chainermn.communicators.MpiCommunicatorBase.gather')

        is_master = self.mpi_comm.rank == root

        msgtype = _MessageType(x)
        _check_dtype('gather', msgtype)

        msgtypes = self.mpi_comm.gather(msgtype, root)

        if is_master:
            _check_dtypes_are_same(msgtypes)

            for msgtype in msgtypes:
                if msgtype.is_tuple:
                    raise TypeError('gather cannot handle tuple data')

                assert len(msgtype.shapes) == 1

            sbuf = _memory_utility.array_to_buffer_object(x)
            shapes = [mty.shapes[0] for mty in msgtypes]
            rlens = [numpy.prod(s) for s in shapes]
            rbuf = numpy.empty(sum(rlens), dtype=msgtype.dtype)

            if chainer.cuda.get_array_module(x) is not numpy:
                chainer.cuda.Stream.null.synchronize()

            self.mpi_comm.Gatherv(
                sbuf,
                [rbuf, (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(msgtype)],
                root)

            ys = [rbuf[i:i + l].reshape(s)
                  for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]
            return tuple(ys)

        else:
            sbuf = _memory_utility.array_to_buffer_object(x)
            self.mpi_comm.Gatherv(sbuf, None, root)
            return None

    def allgather(self, x):
        chainer.utils.experimental(
            'chainermn.communicators.MPICommunicatorBase.allgather')

        msgtype = _MessageType(x)
        _check_dtype('allgather', msgtype)

        msgtypes = self.mpi_comm.allgather(msgtype)
        _check_dtypes_are_same(msgtypes)

        # Type check.
        for msgtype in msgtypes:
            if msgtype.is_tuple:
                raise TypeError('allgather cannot handle tuple data')

            assert len(msgtype.shapes) == 1

        # Collective communication.
        xp = chainer.cuda.get_array_module(x)
        shapes = [msgtype.shapes[0] for msgtype in msgtypes]
        sbuf = _memory_utility.array_to_buffer_object(x)
        rlens = [numpy.prod(s) for s in shapes]
        rbuf = numpy.empty(sum(rlens), dtype=msgtype.dtype)
        if xp is not numpy:
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Allgatherv(
            sbuf,
            [rbuf, (rlens, _cnt_to_dsp(rlens)), _get_mpi_type(msgtype)])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    def allreduce(self, x):
        """A primitive of inter-process allreduce communication.

        This method tries to invoke allreduce communication within the
        communicator. All processes in the communicator are expected to
        invoke ``allreduce()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        Note that this method can only handle the same shapes of data
        over all processes, and cannot handle tuple data.

        Args:
            x (numpy.array): An array to apply allreduce operation.

        Returns:
            ys (numpy.ndarray): An array that allreduce (currently SUM only)
                has been applied.

        """

        chainer.utils.experimental(
            'chainermn.communicators.CommunicatorBase.allreduce')

        msgtype = _MessageType(x)
        _check_dtype('allreduce', msgtype)

        if msgtype.is_tuple:
            raise TypeError('allreduce cannot handle tuple data')

        # TODO(kuenishi): do we check all messages have same shape and dims?

        # Source buffer
        sbuf = _memory_utility.array_to_buffer_object(x)
        # Destination buffer
        dbuf = numpy.empty(msgtype.shapes[0], dtype=msgtype.dtype)
        self.mpi_comm.Allreduce(sbuf, dbuf)

        return dbuf.reshape(msgtype.shapes[0])

    # Objects
    def send_obj(self, obj, dest):
        self.mpi_comm.send(obj, dest=dest)

    def recv_obj(self, source):
        return self.mpi_comm.recv(source=source)

    def bcast_obj(self, obj, max_buf_len=256 * 1024 * 1024, root=0):
        return chunked_bcast_obj(obj, self.mpi_comm,
                                 max_buf_len=max_buf_len,
                                 root=root)

    def gather_obj(self, obj, root=0):
        return self.mpi_comm.gather(obj, root=root)

    def scatter(self, xs, root=0):
        """A primitive of inter-process scatter communication.

        This method tries to invoke scatter communication within the
        communicator. All processes in the communicator are expected to
        invoke ``scatter()``. This method relies on mpi4py fast communication
        optimized for numpy arrays, as well as ``send()`` and ``recv()``.

        If ``xs`` is tuple, each element is send to different processes.
        The length of the tuple must be the same as the communicator size.
        If ``xs`` is ``numpy.ndarrray``, it is splitted with the first
        axis and sent to different processes. For slave processes, ``xs``
        is allowed to be any value (will be ignored).

        Args:
            xs (tuple of numpy.array or numpy.array): Arrays to be scattered.
            root (int): Rank of root process.

        Returns:
            ys (numpy.ndarray): Received arrays.
        """
        chainer.utils.experimental(
            'chainermn.communicators.CommunicatorBase.scatter')

        is_master = self.mpi_comm.rank == root

        if is_master:
            # Type check.
            msgtype = _MessageType(xs)
            _check_dtype('scatter', msgtype)

            if msgtype.is_tuple:
                if len(msgtype.shapes) != self.size:
                    raise ValueError(
                        'the length of xs must be consistent '
                        'with communicator size')

                xp = chainer.cuda.get_array_module(*xs)
                msgtype = tuple([_MessageType(x) for x in xs])
                shapes = [mty.shapes[0] for mty in msgtype]
                # concatenate([x.reshape(-1) ... ], axis=0) will fail
                xs = xp.concatenate([x.reshape(1, -1) for x in xs], axis=1)

            else:
                assert len(msgtype.shapes) == 1

                if msgtype.shapes[0][0] != self.mpi_comm.size:
                    raise ValueError(
                        'scatter received inconsistent number of inputs '
                        'with communicator size')

                xp = chainer.cuda.get_array_module(xs)
                msgtype = tuple([_MessageType(xs[0])
                                 for _ in range(self.size)])
                shapes = [xs.shape[1:] for _ in range(self.size)]

            msgtype = self.mpi_comm.scatter(msgtype, root)
            shape = msgtype.shapes[0]

            # Collective communication.
            slens = [numpy.prod(s) for s in shapes]
            sbuf = _memory_utility.array_to_buffer_object(xs)[0]
            rbuf = numpy.empty(numpy.prod(shape), dtype=msgtype.dtype)
            if xp is not numpy:
                chainer.cuda.Stream.null.synchronize()

            self.mpi_comm.Scatterv(
                [sbuf, (slens, _cnt_to_dsp(slens)), _get_mpi_type(msgtype)],
                rbuf, root)

            return rbuf.reshape(shape)

        else:  # slave processes
            msgtype = self.mpi_comm.scatter(None, root)
            shape = msgtype.shapes[0]
            rbuf = numpy.empty(numpy.prod(shape), dtype=msgtype.dtype)
            self.mpi_comm.Scatterv(None, rbuf, root)
            return rbuf.reshape(shape)

    def allreduce_obj(self, obj):
        # Summation by default
        return self.mpi_comm.allreduce(obj)

    def bcast_data(self, model):
        for _, param in sorted(model.namedparams()):
            buf = _memory_utility.array_to_buffer_object(param.data)
            self.mpi_comm.Bcast(buf)

    # Private methods
    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self._intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]
