import torch.nn as nn
import torchtext
from torchtext.data import get_tokenizer


tokenizer = get_tokenizer("basic_english")
tokens = tokenizer("You can now install TorchText using pip!")
print(tokens)


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
    def __init__(self):
        super(DecoderLayer, self).__init__()
        pass

    def foward(self, x :Tensor):
        pass

class EncoderLayer(nn.Module):
    ''' Transform input tokens into contextualized representations '''
    def __init__(self, input_tokens, embed_dim, num_heads, dropout=0.2):
        super(EncoderLayer, self).__init__()
        self.input_tokens = input_tokens
        self.pos_embeddings = []
        self.ffn = []
        #self.embedding = nn.Embedding(n, d, max_norm=1.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.unembedding = nn.Softmax(dim=1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        x = self.norm1(x + self.dropout(attn_output))
        return attn_output, attn_output_weights
