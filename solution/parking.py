"""What the backward needs of a layer waits on the host until its turn.

The reverse sweep needs, per layer, the outputs of G and U, the fp16 input, the dropout masks, and
the q, k, v and output of the attention. Rerunning what it can rerun is 24% of the multiplies of a
step, and multiplies are what a T4 at its 70 W cap runs out of. Keeping those outputs on the GPU
takes 51 KB per token and layer, more than the whole peak; the inputs alone take 115 MiB of it on
the largest batch. Pinned host memory takes all of it at 11 GiB/s on the GPU's copy engine, beside
the compute. Parking G and U against rerunning them, in one session on a T4: the loop went from
35.5 to 30.8 s and the whole run from 52.1 to 48.7 s, with 1.2 s more outside the loop, 0.4 s of it
for unpinning at exit.

There is one slot on the GPU for G and U and two each for the rest, so that a layer never writes
over the copy of the layer before it, and the copies run on a stream of their own. The forward
fills the slots and sends them while the main stream goes on; the backward fetches the next layer
as soon as the main stream has finished with the slot of G and U, while it works through the rest
of the layer, so neither side waits unless a batch is long enough for a copy to outrun the layer.
Idea of offloading activations: Soup's `activation_offloading`, a switch this task forbids and which
stays off; the code here is ours.
"""

import torch

from .startup import own_stream

PIECES = 5  # rows of the attention slot per token, `hidden` wide: q, k, v, y and h before D


class Parking:
    """The host tensors are pinned, one per layer that waits, [tokens of the largest batch, ...],
    and the masks have two sites in front of that. The last layer never waits: it is in its slots
    when the backward comes. Host tensors are read by contiguous rows only: a non_blocking copy of
    anything else would go through a temporary on the CPU, at once, without waiting for the
    streams."""

    def __init__(self, host_both, host_x, host_bits, host_attention, device, keep_first=False):
        self.host_both, self.host_x, self.host_bits = host_both, host_x, host_bits
        self.host_attention = host_attention
        self.layers, self.device = len(host_both), device
        self.main = torch.cuda.current_stream(device)
        self.copies = own_stream(device)
        self.arrived = None
        self.keep_first = keep_first
        self.retained = None

    def start_step(self, tokens):
        self.tokens = tokens
        self.retained = None
        self.both = self.empty(self.host_both[0])
        self.x = [self.empty(self.host_x[0]) for _ in range(2)]
        self.bits = [[self.empty(self.host_bits[0][0]) for _ in range(2)] for _ in range(2)]
        self.attention = [self.empty(self.host_attention[0], PIECES * tokens) for _ in range(2)]

    def empty(self, host, rows=None):
        return torch.empty(rows or self.tokens, host.shape[-1], dtype=host.dtype,
                           device=self.device)

    def end_step(self):
        self.both, self.x, self.bits, self.attention = None, None, None, None
        self.arrived = None
        self.retained = None

    def wait(self):
        """The main stream may use the slot of G and U once every copy in flight is done. That
        also frees the slots of the layer before, which the next layer writes."""
        self.main.wait_stream(self.copies)

    def slots(self, layer):
        """The input, the masks and the attention of a layer."""
        return self.x[layer % 2], self.bits[layer % 2], self.attention[layer % 2]

    def inputs(self, layer):
        """The slots of a layer, once they have arrived."""
        if self.arrived is not None:
            self.main.wait_event(self.arrived)
            self.arrived = None
        return self.slots(layer)

    def send(self, layer):
        """The slots to the host, after what the main stream has queued so far."""
        if layer == 0 and self.keep_first:
            self.retained = (self.both.clone(), self.x[0].clone(),
                             [part[0].clone() for part in self.bits], self.attention[0].clone())
            return
        tokens = self.tokens
        with torch.cuda.stream(self.copies):
            self.copies.wait_stream(self.main)
            self.host_x[layer][:tokens].copy_(self.x[layer % 2], non_blocking=True)
            for site, bits in enumerate(self.bits[layer % 2]):
                self.host_bits[site][layer][:tokens].copy_(bits, non_blocking=True)
            self.host_attention[layer][:PIECES * tokens].copy_(self.attention[layer % 2],
                                                               non_blocking=True)
            self.host_both[layer][:tokens].copy_(self.both, non_blocking=True)

    def fetch(self, layer):
        """A layer back into its slots, after what the main stream has queued so far."""
        if layer == 0 and self.keep_first:
            self.both, self.x[0], saved_bits, self.attention[0] = self.retained
            for site, bits in enumerate(saved_bits):
                self.bits[site][0] = bits
            self.retained = None
            return
        tokens = self.tokens
        with torch.cuda.stream(self.copies):
            self.copies.wait_stream(self.main)
            self.x[layer % 2].copy_(self.host_x[layer][:tokens], non_blocking=True)
            for site, bits in enumerate(self.bits[layer % 2]):
                bits.copy_(self.host_bits[site][layer][:tokens], non_blocking=True)
            self.attention[layer % 2].copy_(self.host_attention[layer][:PIECES * tokens],
                                            non_blocking=True)
            self.arrived = self.copies.record_event()
            self.both.copy_(self.host_both[layer][:tokens], non_blocking=True)
