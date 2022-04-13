import torch
import torch.nn.functional as F


class RBM:
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        device: str = "cpu",
        learning_rate: float = 0.1,
    ) -> None:
        """
        Construct the RBM model with given number of visible and hidden units

        :arg visible_units: number of visible units
        :arg hidden_units: number of hidden units
        :arg device: the device to instantiate the tensor on
        :arg learning_rate: rate at which to modify weights and biases
        """
        self.device = device
        self.learning_rate = learning_rate
        self.w = torch.randn(n_hidden, n_visible, 5, device=self.device)

        self.v_bias = torch.randn(n_visible, 5, device=self.device)
        self.h_bias = torch.randn(n_hidden, device=self.device)

    def sample_h(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sample hidden units given that v is the visible layer
        :param v: visible layer
        """

        a = torch.sum(torch.matmul(self.w, v.t()), dim=[1, 2])

        activation = self.h_bias + a

        p_h_given_v = torch.sigmoid(activation)
        # print(torch.bernoulli(p_h_given_v))
        return p_h_given_v, torch.sigmoid(p_h_given_v)

    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given that h is the hidden layer
        :param h: hidden layer
        """

        hw = torch.matmul(self.w.permute(1, 2, 0), h.t())
        # print(hw.shape)
        top = torch.exp(hw + self.v_bias.expand_as(hw))
        bottom = torch.sum(top, dim=1)
        # print(bottom.shape)
        p_v_given_h = torch.div(top.t(), bottom).t()
        return p_v_given_h, F.gumbel_softmax(p_v_given_h, hard=True)

    def train(self, goodSample: torch.Tensor, badSample: torch.Tensor) -> None:
        """
        Perform contrastive divergence algorithm to optimize the weights that minimize the energy
        This maximizes the log-likelihood of the model
        """

        good_h = self.sample_h(goodSample)[1]
        bad_h = self.sample_h(badSample)[1]

        # caluclate the deltas
        hb_delta = good_h - bad_h
        vb_delta = goodSample - badSample
        # print(self.w.shape, hb_delta.shape)

        w_delta = (self.w.permute(1, 2, 0) * hb_delta).permute(2, 0, 1)
        # print(w_delta.shape, vb_delta.shape)
        w_delta = w_delta * vb_delta
        # print(w_delta.shape)
        # update the parameters of the model
        self.v_bias += vb_delta * self.learning_rate
        self.h_bias += hb_delta * self.learning_rate
        self.w += w_delta * self.learning_rate

    def reconstruct(self, v: torch.Tensor) -> torch.Tensor:
        """
        For a given v input tensor, let the RBM reconstruct it
        by performing a forward and backward pass
        :arg v: the input tensor
        """

        return self.sample_v(self.sample_h(v)[1])[1]


if __name__ == "__main__":
    rbm = RBM(100, 20)
    rbm.reconstruct(None)
