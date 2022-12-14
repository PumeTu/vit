import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchifyEmbed(nn.Module):
    '''
    Split an image into patches and embed the patches
    Args:
        img_size (int): size of the image 
        patch_size (int): size of the patch 
        in_channels (int): number of input channels
        embed_dim (int): embedding dimension
    '''
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patchify = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.patchify(x)
        x = x.flatten(start_dim=-2, end_dim=-1)
        x = x.transpose(1, 2)
        return x

class SelfAttention(nn.Module):
    '''
    Vanilla Self Attention Block
    '''
    def __init__(self, dim, n_head=12, qkv_bias=True, atten_pdrop=0., proj_pdrop=0.):
        self.n_head = n_head
        self.dim = dim
        #queries, key, value projection
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        #output projection
        self.proj = nn.Linear(dim, dim)

        #Regularization
        self.attn_drop = nn.Dropout(atten_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

    def forward(self, x):
        B, T, C = x.size() #batch size, sequence length, embedding dimensionality

        q, k ,v  = self.qkv(x).split(self.dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, n_heads, mlp_ratio, qkv_bias, proj_pdrop=0., atten_pdrop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = SelfAttention(dim, n_head=n_heads, qkv_bias=qkv_bias, atten_pdrop=atten_pdrop, proj_pdrop=proj_pdrop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x)) 
        x = x + self.mlp(self.norm2(x))

        return x

class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=10, embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True, proj_pdrop=0., atten_pdrop=0.):
        super().__init__()

        self.patch_embed = PatchifyEmbed(img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=proj_pdrop)

        self.blocks = nn.ModuleList([Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, proj_pdrop=proj_pdrop, atten_pdrop=atten_pdrop) for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        class_token = self.class_token.expand(n_samples, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        class_token_final = x[:, 0]
        x = self.head(class_token_final)

        return x