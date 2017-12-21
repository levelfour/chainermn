import chainer
import chainer.functions.connection as conn
import chainermn.functions


_rnn_n_cells = {
    conn.n_step_gru.n_step_bigru: 1,
    conn.n_step_gru.n_step_gru: 1,
    conn.n_step_lstm.n_step_bilstm: 2,
    conn.n_step_lstm.n_step_lstm: 2,
    conn.n_step_rnn.n_step_birnn: 1,
    conn.n_step_rnn.n_step_rnn: 1,
}


class _MultiNodeNStepRNN(chainer.Chain):

    def __init__(self, link, communicator, rank_in, rank_out):
        super(_MultiNodeNStepRNN, self).__init__(actual_rnn=link)

        self.communicator = communicator
        self.rank_in = rank_in
        self.rank_out = rank_out

        if not hasattr(link, 'rnn') or link.rnn not in _rnn_n_cells:
            raise ValueError('link must be NStepRNN and its inherited link')
        else:
            self.n_cells = _rnn_n_cells[link.rnn]

    def recv_states(self):
        cells = [None for _ in range(self.n_cells)]

        if self.rank_in is not None:
            cells = [chainermn.functions.recv(
                self.communicator,
                rank=self.rank_in,
                device=self.actual_rnn._device_id)
                for _ in range(self.n_cells)]

        return cells

    def send_states(self, *states):
        # This utility can only be used for test phase.
        for state in states:
            chainermn.functions.send(
                state, self.communicator, rank=self.rank_out)

    def __call__(self, *inputs):
        cells = self.recv_states()
        outputs = self.actual_rnn(*(tuple(cells) + inputs))
        cells = outputs[:-1]  # cell statuses except last elements
        output = outputs[-1]

        delegate_variable = None
        if self.rank_out is not None:
            cell = cells[0]
            for i in range(self.n_cells):
                delegate_variable = chainermn.functions.send(
                    cell, self.communicator, rank=self.rank_out)
                if i < self.n_cells - 1:
                    cell = chainermn.functions.pseudo_connect(
                        delegate_variable, cells[i + 1])
                else:
                    output = chainermn.functions.pseudo_connect(
                        delegate_variable, *output)

        return cells + tuple([output, delegate_variable])


def create_multi_node_n_step_rnn(
        actual_link, communicator, rank_in=None, rank_out=None):
    """Create a multi node stacked RNN link from a Chainer stacked RNN link.

    Multi node stacked RNN link is used for model-parallel.
    The created link will receive initial hidden states from the process
    specified by ``rank_in`` (or do not receive if ``None``), execute
    the original RNN compuation, and then send resulting hidden states
    to the process specified by ``rank_out``.

    Compared with Chainer stacked RNN link, multi node stacked RNN link
    returns an extra object called ``delegate_variable``.
    If ``rank_out`` is not ``None``, backward computation is expected
    to be begun from ``delegate_variable``.
    For detail, please refer ``chainermn.functions.pseudo_connect``.

    The following RNN links can be passed to this function:

    - ``chainer.links.NStepBiGRU``
    - ``chainer.links.NStepBiLSTM``
    - ``chainer.links.NStepBiRNNReLU``
    - ``chainer.links.NStepBiRNNTanh``
    - ``chainer.links.NStepGRU``
    - ``chainer.links.NStepLSTM``
    - ``chainer.links.NStepRNNReLU``
    - ``chainer.links.NStepRNNTanh``

    Args:
        link (chainer.Link): Chainer stacked RNN link
        communicator: ChainerMN communicator
        rank_in (int, or None):
            Rank of the process which sends hidden RNN states to this process.
        rank_out (int, or None):
            Rank of the process to which this process sends hiddne RNN states.

    Returns:
        The multi node stacked RNN link based on ``actual_link``.
    """
    chainer.utils.experimental('chainermn.links.create_multi_node_n_step_rnn')
    return _MultiNodeNStepRNN(actual_link, communicator, rank_in, rank_out)
