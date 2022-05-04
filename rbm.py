from random import sample
import torch
import torch.nn.functional as F
from utils import *


class RBM:
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        device: str = "cpu",
        learning_rate: float = 0.0001,
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

        self.w = torch.zeros(n_hidden, n_visible, 5, device=self.device)
        self.v_bias = torch.randn(n_visible, 5, device=self.device)
        self.h_bias = torch.randn(n_hidden, device=self.device) * -4

        self.prev_w_delta = torch.zeros(n_hidden, n_visible, 5, device=self.device)
        self.prev_vb_delta = torch.zeros(n_visible, 5, device=self.device)
        self.prev_hb_delta = torch.zeros(n_hidden, device=self.device)

        self.adaptive = adaptive
        self.momentum = momentum

    def sample_h(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sample hidden units given that v is the visible layer
        :param v: visible layer
        """

        a = torch.sum(torch.matmul(self.w, v.t()), dim=[1, 2])

        activation = self.h_bias + a

        phv = torch.sigmoid(activation)

        return phv, torch.bernoulli(phv)

    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given that h is the hidden layer
        :param h: hidden layer
        """

        hw = torch.matmul(self.w.permute(1, 2, 0), h.t())

        pvh = hw + self.v_bias.expand_as(hw)

        pvh = pvh.softmax(dim=1)

        return pvh, softmax_to_onehot(pvh)

    def apply_gradient(self, minibatch: torch.Tensor, t: int = 1) -> None:
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """
        hb_delta = torch.zeros(self.n_hidden, device=self.device)
        vb_delta = torch.zeros(self.n_visible, 5, device=self.device)
        w_delta = torch.zeros(self.n_hidden, self.n_visible, 5, device=self.device)

        for case in minibatch:
            v0 = case
            pvk = case

            _, h0 = self.sample_h(v0)
            phk = h0

            # do gibbs sampling for t steps
            for i in range(t):
                phk, hk = self.sample_h(pvk)  # forward pass
                pvk, vk = self.sample_v(phk)  # backward pass

                pvk[v0.sum(dim=1) == 0] = v0[v0.sum(dim=1) == 0]
                vk[v0.sum(dim=1) == 0] = v0[v0.sum(dim=1) == 0]

            phk, hk = self.sample_h(pvk)

            # caluclate the deltas
            hb_delta += h0 - hk
            vb_delta += v0 - vk

            w_delta += hb_delta.view([self.n_hidden, 1, 1]) * vb_delta.view(
                [1, self.n_visible, 5]
            )

        # apply momentum if applicable
        w_delta += self.momentum * self.prev_w_delta
        hb_delta += self.momentum * self.prev_hb_delta
        vb_delta += self.momentum * self.prev_vb_delta

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

        return self.sample_v(self.sample_h(v)[1])[1]
