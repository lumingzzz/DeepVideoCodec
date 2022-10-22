import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.utils import update_registered_buffers
from .entropy_models import DiscretizedGaussian, DiscretizedLaplace


class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.
    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    def __init__(self, y_distribution, z_channel, mv_z_channel=None):
        super().__init__()
        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.mv_z_channel = mv_z_channel
        self.entropy_coder = None
        self.bit_estimator_z = EntropyBottleneck(z_channel)
        self.bit_estimator_z_mv = None
        if mv_z_channel is not None:
            self.bit_estimator_z_mv = EntropyBottleneck(mv_z_channel)
        self.bit_estimator_y = None
        if y_distribution == 'gaussian':
            self.bit_estimator_y = DiscretizedGaussian()
        elif y_distribution == 'laplace':
            self.bit_estimator_y = DiscretizedLaplace()

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def quant(self, x, force_detach=False):
        if self.training or force_detach:
            n = torch.round(x) - x
            n = n.clone().detach()
            return x + n

        return torch.round(x)

    def forward(self, *args):
        raise NotImplementedError()

    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.
        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.
        Args:
            force (bool): overwrite previous values (default: False)
        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.
        """
        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        super().load_state_dict(state_dict)