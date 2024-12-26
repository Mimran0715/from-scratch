import torch.nn as nn
import torchtext
from torchtext.data import get_tokenizer
import torch
import torch.optim as optim
import torch.utils.data as data
import math
import copy

# implementation taken from PyTorch Documentation 
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# end

class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dff, dropout=0.2):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = []
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class EncoderLayer(nn.Module):
    ''' Transform input tokens into contextualized representations '''
    def __init__(self, embed_dim, num_heads, dff, input_tokens,  dropout=0.2,):
        super(EncoderLayer, self).__init__()
        self.input_tokens = input_tokens
        self.pos_embeddings = []
        self.feed_forward = []
        self.dff = dff
        #self.embedding = nn.Embedding(n, d, max_norm=1.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.unembedding = nn.Softmax(dim=1)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    # def forward(self, query, key, value):
    #     attn_output, attn_output_weights = self.multihead_attn(query, key, value)
    #     x = self.norm1(x + self.dropout(attn_output))
    #     return attn_output, attn_output_weights

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

def test_transformer():

    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer("You can now install TorchText using pip!")
    print(tokens)


if __name__ == "__main__":
    test_transformer()