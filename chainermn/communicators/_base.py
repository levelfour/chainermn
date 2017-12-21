import collections

import mpi4py
import numpy

import chainer.cuda
import chainer.utils
from chainermn.communicators import _communication_utility
from chainermn.communicators import _memory_utility
from chainermn import nccl


MSGTYPE_OPTIMIZATION = False
MPITAG_MSGTYPE = 1 << 8


def _cnt_to_dsp(cnt):
    """Utility to convert length array to cumulative array."""
    return [0] + numpy.cumsum(cnt)[:-1].tolist()


class _MessageType(object):

    def __init__(self, obj):
        if isinstance(obj, numpy.ndarray) \
                or chainer.cuda.get_array_module(obj) is not numpy:
            self.is_tuple = False
            self.narr = 1
            self.ndims = [obj.ndim]
            self.shapes = [obj.shape]
        elif isinstance(obj, collections.Iterable):
            self.is_tuple = True
            self.narr = len(obj)
            self.ndims = [x.ndim for x in obj]
            self.shapes = [x.shape for x in obj]
        elif obj is None:
            self.is_tuple = None
            self.narr = None
            self.ndims = None
            self.shapes = None
        else:
            raise ValueError(
                'Message object must be numpy/cupy array or tuple.')


class CommunicatorBase(object):

    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm

    @property
    def rank(self):
        return self.mpi_comm.rank

    @property
    def size(self):
        return self.mpi_comm.size

    def split(self, color, key):
        """A wrapper function of MPI_Comm_Split.

        This method splits the inter MPI commnicator and return a wrapped
        ChainerMN communicator.

        Args:
            color (int):
                Index of new group. The process with the same color will be
                assigned to the same group.
            key (int):
                Control of rank assignment. The process will be assigned
                a rank in the new group ordered by the value of key.
                If you do not care of the rank, you can just simply specify
                the original rank.

        Returns:
            CommunicatorBase
        """
        return self.__class__(mpi_comm=self.mpi_comm.Split(color, key))

    def send(self, obj, dest, tag):
        """A primitive for inter-process transmitter.

        This method sends numpy-array to target process.
        The target process is expected to invoke ``recv()``.
        This method relies on mpi4py fast communication optimized for
        numpy arrays, which discards any information attached to
        chainer.Variable objects. Please be sure.

        Args:
            obj: data to be sent (tuple, list or raw numpy/cupy array)
            dest (int): Target process specifier.
            tag (int): Message ID (MPI feature).

        """
        chainer.utils.experimental(
            'chainermn.communicators.CommunicatorBase.send')

        msgtype = self._send_msgtype(obj, dest=dest)

        if not msgtype.is_tuple:
            obj = [obj]

        for array in obj:
            if chainer.cuda.get_array_module(array) is not numpy:
                chainer.cuda.Stream.null.synchronize()

            buf = _memory_utility.array_to_buffer_object(array)
            self.mpi_comm.Send(buf, dest=dest, tag=tag)

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
            'chainermn.communicators.CommunicatorBase.recv')

        msgtype = self._recv_msgtype(source=source)

        if msgtype.is_tuple:
            msg = []
            for shape in msgtype.shapes:
                buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
                self.mpi_comm.Recv(buf, source=source, tag=tag)
                msg.append(buf.reshape(shape))
            return tuple(msg)

        else:
            assert len(msgtype.shapes) == 1
            shape = msgtype.shapes[0]
            buf = numpy.empty(numpy.prod(shape), dtype=numpy.float32)
            self.mpi_comm.Recv(buf, source=source, tag=tag)
            return buf.reshape(shape)

    def _send_msgtype(self, obj, dest):
        msgtype = _MessageType(obj)

        if MSGTYPE_OPTIMIZATION:
            reduced_msgtype = []
            reduced_msgtype.append(1 if msgtype.is_tuple else 0)
            reduced_msgtype.append(msgtype.narr)
            reduced_msgtype += msgtype.ndims
            for shape in msgtype.shapes:
                reduced_msgtype += list(shape)
            reduced_msgtype = numpy.array(reduced_msgtype, dtype=numpy.int32)
            buf = [reduced_msgtype, mpi4py.MPI.INT]
            self.mpi_comm.Send(buf, dest=dest, tag=MPITAG_MSGTYPE)
        else:
            self.mpi_comm.send(msgtype, dest=dest, tag=MPITAG_MSGTYPE)

        return msgtype

    def _recv_msgtype(self, source):
        if MSGTYPE_OPTIMIZATION:
            stat = mpi4py.MPI.Status()
            self.mpi_comm.Probe(source=source, tag=MPITAG_MSGTYPE, status=stat)
            buflen = stat.Get_count(mpi4py.MPI.INT)
            buf = numpy.empty(buflen, dtype=numpy.int32)
            self.mpi_comm.Recv(
                [buf, mpi4py.MPI.INT], source=source, tag=MPITAG_MSGTYPE)
            msgtype = _MessageType(None)
            msgtype.is_tuple = buf[0] == 1
            msgtype.narr = buf[1]
            msgtype.ndims = buf[2:2 + msgtype.narr]
            msgtype.shapes = []
            i = 2 + msgtype.narr
            for n in range(msgtype.narr):
                shape = tuple(buf[i:i + msgtype.ndims[n]])
                msgtype.shapes.append(shape)
                i += msgtype.ndims[n]
            return msgtype
        else:
            return self.mpi_comm.recv(source=source, tag=MPITAG_MSGTYPE)

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
            'chainermn.communicators.CommunicatorBase.all_to_all')

        if len(xs) != self.size:
            raise ValueError(
                'The length of data must be same as communicator size.')

        # Mediate #axes of arrays.
        sndims = numpy.array([x.ndim for x in xs], dtype=numpy.int32)
        rndims = numpy.empty(self.size, dtype=numpy.int32)
        self.mpi_comm.Alltoall(
            [sndims, mpi4py.MPI.INT],
            [rndims, mpi4py.MPI.INT])

        # Arbitrate shapes of arrays.
        sshapes = numpy.hstack([x.shape for x in xs]).astype(numpy.int32)
        rshapes = numpy.empty(sum(rndims), dtype=numpy.int32)
        self.mpi_comm.Alltoallv(
            [sshapes, (sndims, _cnt_to_dsp(sndims)), mpi4py.MPI.INT],
            [rshapes, (rndims, _cnt_to_dsp(rndims)), mpi4py.MPI.INT])
        shapes = [rshapes[i:i + l]
                  for i, l in zip(_cnt_to_dsp(rndims), rndims)]

        # Collective communication.
        slens = [numpy.prod(x.shape) for x in xs]
        xp = chainer.cuda.get_array_module(xs[0])
        sbuf = xp.hstack([x.reshape(-1) for x in xs])
        rlens = [numpy.prod(s) for s in shapes]
        rbuf = numpy.empty(sum(rlens), dtype=numpy.float32)
        if xp is not numpy:
            sbuf = _memory_utility.array_to_buffer_object(sbuf)[0]
            chainer.cuda.Stream.null.synchronize()
        self.mpi_comm.Alltoallv(
            [sbuf, (slens, _cnt_to_dsp(slens)), mpi4py.MPI.FLOAT],
            [rbuf, (rlens, _cnt_to_dsp(rlens)), mpi4py.MPI.FLOAT])
        ys = [rbuf[i:i + l].reshape(s)
              for i, l, s in zip(_cnt_to_dsp(rlens), rlens, shapes)]

        return tuple(ys)

    def broadcast_data(self, model):
        raise NotImplementedError()

    def allreduce_grad(self, model):
        raise NotImplementedError()


class NodeAwareCommunicatorBase(CommunicatorBase):

    def __init__(self, mpi_comm, use_nccl):
        super(NodeAwareCommunicatorBase, self).__init__(mpi_comm)

        if use_nccl and not nccl._available:
            raise RuntimeError(
                'NCCL is not available. '
                'Please confirm that NCCL can be found by dynamic linkers, '
                'and ChainerMN is installed without --no-nccl flag.'
            )

        self.use_nccl = use_nccl

        self._init_ranks()

        # We have to delay the initialization of communicators. This is because
        # NCCL's communicators use the current CUDA devices at the time of
        # initialization. Therefore, we have to initialize NCCL communicators
        # after users set the devices to use.
        self.inter_mpi_comm = None
        self.intra_mpi_comm = None
        if self.use_nccl:
            self.intra_nccl_comm = None

    def _init_ranks(self):
        my_ranks = _communication_utility.init_ranks(self.mpi_comm)
        assert my_ranks[0] == self.mpi_comm.rank
        self.intra_rank = my_ranks[1]
        self.intra_size = my_ranks[2]
        self.inter_rank = my_ranks[3]
        self.inter_size = my_ranks[4]

    def _init_comms(self):
        if self.inter_mpi_comm is not None:
            assert self.intra_mpi_comm is not None
            assert not self.use_nccl or self.intra_nccl_comm is not None
            return

        comms = _communication_utility.init_comms(
            self.mpi_comm, self.intra_rank, self.intra_size, self.inter_rank,
            use_nccl=self.use_nccl)
        self.intra_mpi_comm = comms[0]
        self.inter_mpi_comm = comms[1]
        if self.use_nccl:
            self.intra_nccl_comm = comms[2]
