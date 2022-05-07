from typing import Tuple
import torch
import torch.nn.functional as F
from utils import *


class RBM:
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        device: str = "cpu",
        learning_rate: float = 0.001,
        adaptive=False,
        momentum=0.0,
    ) -> None:
        """
        Construct the RBM model with given number of visible and hidden units

        :arg visible_units: number of visible units
        :arg hidden_units: number of hidden units
        :arg device: the device to instantiate the tensor on
        :arg learning_rate: rate at which to modify weights and biases
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        self.device = device
        self.alpha = learning_rate

        self.w = torch.randn(n_visible, n_hidden, device=self.device)
        self.v_bias = torch.randn(n_visible, device=self.device)
        self.h_bias = torch.randn(1, n_hidden, device=self.device)

        self.prev_w_delta = torch.zeros(n_visible, n_hidden, device=self.device)
        self.prev_vb_delta = torch.zeros(n_visible, device=self.device)
        self.prev_hb_delta = torch.zeros(1, n_hidden, device=self.device)

        self.adaptive = adaptive
        self.momentum = momentum

    def forward_pass(
        self,
        v: torch.Tensor,
        activation=torch.sigmoid,
        sampler=torch.bernoulli,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given that v is the visible layer
        :param v: visible layer
        :param activation: activation function to be used
        :param sampler: sampling function to be used

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: probability and sample tensor
        """
        if len(v.shape) > 1:
            v = v.flatten()
        a = torch.matmul(v, self.w)

        a = self.h_bias + a

        ph = activation(a)

        return ph, sampler(ph)

    def backward_pass(
        self, h: torch.Tensor, activation=ratings_softmax
    ) -> torch.Tensor:
        """
        Sample visible units given that h is the hidden layer
        :param h: hidden layer
        """

        hw = torch.matmul(h, self.w.t())

        pv = self.v_bias + hw
        pv = activation(pv.flatten())
        return pv

    def apply_gradient(self, minibatch: torch.Tensor, t: int = 1) -> None:
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """
        vb_delta = torch.zeros(self.n_visible, device=self.device)
        hb_delta = torch.zeros(1, self.n_hidden, device=self.device)
        w_delta = torch.zeros(self.n_visible, self.n_hidden, device=self.device)

        for case in minibatch:
            v0 = case

            _, h0 = self.forward_pass(v0)
            hk = phk = h0

            # do Gibbs sampling for t steps
            for i in range(t):
                vk = self.backward_pass(hk)  # backward pass

                phk, hk = self.forward_pass(vk)  # forward pass
            vk[v0.sum(dim=1) == 0] = v0[v0.sum(dim=1) == 0]  # remove missing
            # flatten v0 and vk
            v0 = v0.flatten()
            vk = vk.flatten()

            # caluclate the deltas
            hb_delta += h0 - hk
            vb_delta += v0 - vk

        w_delta = (vb_delta * hb_delta.t()).t()

        # apply momentum if applicable
        # w_delta += self.momentum * self.prev_w_delta
        # hb_delta += self.momentum * self.prev_hb_delta
        # vb_delta += self.momentum * self.prev_vb_delta

        # divide learning rate by the size of the minibatch
        hb_delta /= len(minibatch)
        vb_delta /= len(minibatch)
        w_delta /= len(minibatch)

        # update the parameters of the model
        self.v_bias += vb_delta * self.alpha
        self.h_bias += hb_delta * self.alpha
        self.w += w_delta * self.alpha

        # remember the deltas for next training step when using momentum
        self.prev_w_delta = w_delta
        self.prev_hb_delta = hb_delta
        self.prev_vb_delta = vb_delta

    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """
        For a given v input tensor, let the RBM reconstruct it
        by performing a forward and backward pass
        :arg v: the input tensor
        """
        _, h = self.forward_pass(v)
        ret = self.backward_pass(h)
        return ret


if __name__ == "__main__":
    model = RBM(100, 20)
    v = torch.randn(20, 5)
    _, h = model.forward_pass(v)
    v = model.backward_pass(h)

    batch = torch.randn(20, 20, 5)
    model.apply_gradient(batch)
