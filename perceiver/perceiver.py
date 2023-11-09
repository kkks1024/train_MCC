from typing import Optional

import torch
import torch.nn as nn

from .encoder import PerceiverEncoder
from .decoder import PerceiverDecoder
from .query_new import Query_Gen_transformer, Query_Gen_transformer_PE
from .query import Query_Gen
#from .query import Query_Gen


class PerceiverIO(nn.Module):

    def __init__(
        self,
        encoder: PerceiverEncoder,
        decoder: PerceiverDecoder,
        query_gen: Query_Gen_transformer_PE,
        decoder_query_dim: int
    ):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.query_gen = query_gen
        #self.query = nn.Parameter(torch.randn(1, decoder_query_dim, decoder_query_dim))

    def forward(
        self,
        inputs: Optional[torch.Tensor],
        query: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None
    ):

        latents = self.encoder(inputs, input_mask)
        query = self.query_gen(inputs)

        outputs = self.decoder(
            x_q=query,
            latents=latents,
            query_mask=query_mask
        )
        return outputs