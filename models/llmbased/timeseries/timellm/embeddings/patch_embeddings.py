import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.padding import ReplicationPad1d

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in,
                                   out_channels=d_model,
                                   kernel_size=3,
                                   padding=padding,
                                   padding_mode='circular',
                                   bias=False)
        self.tokenConv = self.tokenConv.to(torch.float32)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class ReplicationPad1d(nn.Module):
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])
        output = torch.cat([input, replicate_padding], dim=-1)
        return output

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        # The padding layer was initialized as ReplicationPad1d((0, stride)),
        # which means no padding at the beginning of the sequence and stride padding at the end.
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        # self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        x = x.permute(0, 2, 1).contiguous()
        n_vars = x.shape[1] # features
        x_pad = self.padding_patch_layer(x)
        # This operation extracts sliding patches from the tensor along a specified dimension
        # dimension=-1 specifies unfolding along the last dimension (the padded sequence length).
        x_unfolded = x_pad.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # shape will be (batch, n_vars, number_of_patches, patch_len)
        # number of patches = (seq_len - patch_len) // stride +1
        # +1 to account for last patch to be included
        # Reshape the tensor to (batch * n_vars, number_of_patches, patch_len)
        _x = torch.reshape(x_unfolded, (x_unfolded.shape[0] * x_unfolded.shape[1], x_unfolded.shape[2], x_unfolded.shape[3]))
        # Input encoding
        x = self.value_embedding(_x)
        return self.dropout(x), n_vars