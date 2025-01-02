import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from einops import rearrange


# taken from https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27


# class PositionEmbedding(nn.Module):
#     def __init__(self):
#         pass

#     def forward(self, x):
#         return x
    
def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_len):
        for j in range(emb_size):
            embeddings[i][j] = np.sin(i / (pow(10000, j / emb_size))) if j % 2 == 0 else np.cos(i / (pow(10000, (j - 1) / emb_size)))
    return torch.tensor(embeddings)

class PatchEmbedding(nn.Module):

    def __init__(self, input_channels=3,patch_size=16, embedding_size=768, img_size=224):
        super.__init__(self, PatchEmbedding)
        self.patch_size = patch_size
        self.embed = nn.Sequential(
            nn.Conv2D(in_channels=input_channels, out_channels=embedding_size,\
                       kernel_size=patch_size, stride=patch_size), 
            rearrange('b e (h) (w) -> b (h w) e')
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_size))
        self.pos_embed = nn.Parameter(PositionEmbedding((img_size // patch_size)**2 + 1, emb_size)))

    def forward(self, x):
        b, _, _ = x.shape
        x = self.embed(x)
        return x

