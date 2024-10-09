from promptttspp.modules.esp.conformer.encoder import Encoder as _ConformerEncoder
from promptttspp.utils.model import make_non_pad_mask
from torch import nn


def _source_mask(ilens, maxlen=None):
    x_masks = make_non_pad_mask(ilens, maxlen=maxlen).to(ilens.device)
    return x_masks.unsqueeze(-2) & x_masks.unsqueeze(-1)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        idim=8,
        attention_dim=8,
        return_mask=False,
        rel_pos_type=None,
        **kwargs,
    ):
        super(ConformerEncoder, self).__init__()
        self._out_dim = attention_dim
        self.return_mask = return_mask
        if idim == attention_dim:
            input_layer = None
        else:
            input_layer = nn.Linear(idim, attention_dim)

        # Check the relative positional encoding type
        if rel_pos_type is None or rel_pos_type == "legacy":
            # Use legacy relative positional encoding for backward compatibility
            if kwargs.get("pos_enc_layer_type") == "rel_pos":
                kwargs["pos_enc_layer_type"] = "legacy_rel_pos"
            if kwargs.get("selfattention_layer_type") == "rel_selfattn":
                kwargs["selfattention_layer_type"] = "legacy_rel_selfattn"
        elif rel_pos_type != "new":
            raise ValueError(
                f"Unknown relative positional encoding type: {rel_pos_type}"
            )
        self.encoder = _ConformerEncoder(
            idim, attention_dim=attention_dim, input_layer=input_layer, **kwargs
        )

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, emb, input_lens=None):
        """Forward function.
        Args:
            emb (torch.Tensor): Batch of padded embeddings (B, T, idim).
            input_lens (torch.Tensor): Batch of lengths of input sequences (B,).
            spk_emb (torch.Tensor, optional): Batch of speaker embeddings (B, 1, idim).
        Returns:
            torch.Tensor: Batch of padded outputs (B, T, out_dim).
            torch.Tensor: Batch of masks (B, T, 1). Returned only when ``return_mask=True``.
        """
        mask = _source_mask(input_lens, maxlen=emb.shape[1])
        outs = self.encoder(emb, mask)[0]
        # mask out the padding part
        outs = outs * mask[:, :, 0:1].to(outs.dtype)
        if self.return_mask:
            # mask: [b, t, t] -> [b, t, 1]
            mask = mask[:, :, 0:1].to(outs.dtype)
            return outs, mask
        return outs
