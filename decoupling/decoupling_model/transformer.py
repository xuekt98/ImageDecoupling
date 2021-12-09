import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import top_k_top_p_filtering

import pdb


class GPTConfig:
    """
        GPT configuration
    """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class MHSelfAttention(nn.Module):
    """
        MultiHeadSelfAttention 多头注意力模块
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
    
    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
            
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        y = self.resid_drop(self.proj(y))
        return y, present


class TransBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MHSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
    
    def forward(self, x, layer_past=None, return_present=False):
        if return_present: assert not self.training
        
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class GPT(nn.Module):
    """
        GPT模型
    """
    
    def __init__(self, vocab_size, block_size, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0):
        """
        :param vocab_size: token embedding的词典的大小
        :param block_size:
        :param n_layer: Transformer Decoder的层数
        :param n_head: Transformer Decoder的多头注意力的头数
        :param n_embd: token embedding的维度，也是整个计算过程计算的特征维度
        :param embd_pdrop: token embedding之后的dropout
        :param resid_pdrop: 残差连接的dropout
        :param attn_pdrop: 计算注意力的dropout
        :param n_unmasked:
        """
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_unmasked=n_unmasked)
        # input embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # transformer
        self.blocks = nn.Sequential(*[TransBlock(config) for _ in range(config.n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        
    def get_block_size(self):
        return self.block_size
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, embeddings=None, targets=None):
        # forward the GPT model
        # pdb.set_trace()
        token_embeddings = self.tok_emb(idx)
        
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
            token_embeddings = token_embeddings[:, :-1]
        
        t = token_embeddings.shape[1]
        assert t <= self.block_size
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def forward_with_past(self, idx, embeddings=None, targets=None, past=None, past_length=None):
        batch_size = 0
        assert not self.training
        token_embeddings = None
        if idx is not None:
            token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
            batch_size = idx.shape[0]
        
        if embeddings is not None:
            batch_size = embeddings.shape[0]
            if token_embeddings is None:
                token_embeddings = embeddings
            else:
                token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)
        
        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)
            past_shape = list(past.shape)
            expected_shape = [self.config.n_layer, 2, batch_size, self.config.n_head, past_length, self.config.n_embd // self.config.n_head]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :] # each position maps to a (learnable) vector
        else:
            position_embeddings = self.pos_emb[:, :token_embeddings.shape[1], :]
        
        x = self.drop(token_embeddings + position_embeddings)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, layer_past=past[i, ...] if past is not None else None, return_present=True)
            presents.append(present)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, torch.stack(presents)


@torch.no_grad()
def sample_with_past(embeddings, model, steps, sample_logits=True,
                     temperature=1., top_k=None, top_p=None):
    sample = None
    cond_len = embeddings.shape[1]
    past = None
    x = None
    # first = True
    for n in range(steps):
        if x is None:
            # first = True
            logits, _, present = model.forward_with_past(x, embeddings=embeddings, past=past,
                                                         past_length=(n+cond_len-1))
        else:
            # first = False
            logits, _, present = model.forward_with_past(x, embeddings=None, past=past,
                                                         past_length=(n+cond_len-1))

        if past is None:
            past = [present]
        else:
            past.append(present)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(logits, dim=-1)
        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)
        
        if sample is None:
            sample = x
            # print(f'sample={sample}')
        else:
            sample = torch.cat((sample, x), dim=1)
    del past
    return sample




