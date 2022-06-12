from typing import Optional, Tuple

from overrides import overrides
import torch
from torch import nn

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import min_value_of_dtype
from allennlp.nn import util


@Seq2VecEncoder.register("self_attention")
class SelfAttentionEncoder(Seq2VecEncoder):
    """
    # Parameters
    
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: Optional[int] = None,
        num_heads: int = 8
    ) -> None:
        super().__init__()
        
        self._embedding_dim = embedding_dim

        self.multihead_attn = nn.MultiheadAttention(self._embedding_dim, num_heads)
        self._self_attentive_pooling_projection = nn.Linear(embedding_dim, 1)
       
       
        if output_dim:
            self.projection_layer = nn.Linear(embedding_dim, output_dim)
            self._output_dim = output_dim
        else:
            self.projection_layer = None
            self._output_dim = embedding_dim

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._output_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        """

        Args:

        tokens: shape : BTC
        mask: shape : BT
        """
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            # If mask doesn't exist create one of shape (batch_size, num_tokens)
            mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device).bool()

        attn_output, attn_output_weights = self.multihead_attn(tokens, tokens, tokens)
        
        # Self-attentive pooling layer
        # Run through linear projection. Shape: (batch_size, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (batch_size, sequence length).
        self_attentive_logits = self._self_attentive_pooling_projection(
            attn_output
        ).squeeze(2) # shape: BTC -> BT1 -> BT
        self_weights = util.masked_softmax(self_attentive_logits, mask)
        self_attentive_pool = util.weighted_sum(attn_output, self_weights) # BC

        # pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)
        # pooled_representations_dropped = self._integrator_dropout(pooled_representations)

        if self.projection_layer:
            result = self.projection_layer(tokens)
            return result
        else:
            return self_attentive_pool