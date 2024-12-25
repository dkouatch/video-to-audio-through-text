import math
from functools import partial
import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F
from vt2a.modules.custom_transformers import ContinuousTransformerWrapper, EncoderKS
from einops import rearrange, repeat
from vt2a.modules.pos_embed import np_get_1d_sincos_pos_embed
import numpy as np
from vt2a.modules.custom_quantize import ResidualVectorQuantize
from torch.nn.utils import weight_norm
from typing import Optional
from typing import Tuple
import scipy.stats as stats
import random


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    B, T, C = t.shape
    noise_vector = gumbel_noise(t)
    # logits_with_temp = t / max(temperature, 1e-10)
    log_prob = torch.log_softmax(t, dim=dim)
    noisy_logits = log_prob / max(temperature, 1e-10) + noise_vector
    token_probs, tokens = torch.max(noisy_logits, dim=dim)
    # confid_score = log_prob
    # confid_score = confid_score.reshape(B * T, -1)
    # token_probs = confid_score[range(len(confid_score)), tokens.view(-1)].view(tokens.shape)
    # print(token_probs.shape, token_probs)
    # temp = t.reshape(B * T, -1)
    # token_probs = temp[range(len(temp)), tokens.view(-1)].view(tokens.shape)
    return token_probs, tokens

def gumbel_sample_v1(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)


def apply_mask(
        x: torch.Tensor,
        mask: torch.Tensor,
        mask_token: int
):
    assert mask.ndim == 3, "mask must be (batch, n_codebooks, seq), but got {mask.ndim}"
    assert mask.shape == x.shape, f"mask must be same shape as x, but got {mask.shape} and {x.shape}"
    assert mask.dtype == torch.long, "mask must be long dtype, but got {mask.dtype}"
    assert ~torch.any(mask > 1), "mask must be binary"
    assert ~torch.any(mask < 0), "mask must be binary"

    fill_x = torch.full_like(x, mask_token)
    x = x * (1 - mask) + fill_x * mask

    return x, mask


def scalar_to_batch_tensor(x, batch_size):
    return torch.tensor(x).repeat(batch_size)


def _gamma(r):
    return (r * torch.pi / 2).cos().clamp(1e-10, 1.0)

def _gamma_new(r, n=2):
    return torch.pow(1 - torch.pow(r, n), 1 / n).clamp(1e-10, 1.0)


def random_b(
        x: torch.Tensor,
        r: torch.Tensor
):
    assert x.ndim == 3, "x must be (batch, n_codebooks, seq)"
    if not isinstance(r, torch.Tensor):
        r = scalar_to_batch_tensor(r, x.shape[0]).to(x.device)

    # r = _gamma(r)[:, None, None]
    probs = torch.ones_like(x) * r[:, None, None]

    mask = torch.bernoulli(probs)
    mask = mask.round().long()

    return mask

def random_b2(
        x: torch.Tensor,
        r: torch.Tensor
):

    # r = _gamma(r)[:, None, None]
    probs = torch.ones_like(x) * r[:, None, None].to(x.device)

    mask = torch.bernoulli(probs)
    mask = mask.round().long()

    return mask



def codebook_unmask(
    mask: torch.Tensor,
    n_conditioning_codebooks: int
):
    if n_conditioning_codebooks == None:
        return mask
    # if we have any conditioning codebooks, set their mask  to 0
    mask = mask.clone()
    mask[:, :n_conditioning_codebooks, :] = 0
    return mask

def codebook_mask(mask: torch.Tensor, start: int):
    mask = mask.clone()
    mask[:, start:, :] = 1
    return mask

# classifier free guidance functions
def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


# noise schedules
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (int, optional): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def recurse_children(module, fn):
    for child in module.children():
        if isinstance(child, nn.ModuleList):
            for c in child:
                yield recurse_children(c, fn)
        if isinstance(child, nn.ModuleDict):
            for c in child.values():
                yield recurse_children(c, fn)

        yield recurse_children(child, fn)
        yield fn(child)

class SequentialWithFiLM(nn.Module):
    """
    handy wrapper for nn.Sequential that allows FiLM layers to be
    inserted in between other layers.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    @staticmethod
    def has_film(module):
        mod_has_film = any(
            [res for res in recurse_children(module, lambda c: isinstance(c, FiLM))]
        )
        return mod_has_film

    def forward(self, x, cond):
        for layer in self.layers:
            if self.has_film(layer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class FiLM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim > 0:
            self.beta = nn.Linear(input_dim, output_dim)
            self.gamma = nn.Linear(input_dim, output_dim)

    def forward(self, x, r):
        if self.input_dim == 0:
            return x
        else:
            beta, gamma = self.beta(r), self.gamma(r)
            beta, gamma = (
                beta.view(x.size(0), self.output_dim, 1),
                gamma.view(x.size(0), self.output_dim, 1),
            )
            x = x * (gamma + 1) + beta
        return x

class CodebookEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        latent_dim: int,
        n_codebooks: int,
        emb_dim: int,
        special_tokens: Optional[Tuple[str]] = None,
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.emb_dim = emb_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size

        if special_tokens is not None:
            for tkn in special_tokens:
                self.special = nn.ParameterDict(
                    {
                        tkn: nn.Parameter(torch.randn(n_codebooks, self.latent_dim))
                        for tkn in special_tokens
                    }
                )
                self.special_idxs = {
                    tkn: i + vocab_size for i, tkn in enumerate(special_tokens)
                }

        # self.out_proj = nn.ModuleList([nn.Linear(self.latent_dim, self.emb_dim, bias=False) for _ in range(4)])

        # self.out_proj = nn.Conv1d(n_codebooks * self.latent_dim, self.emb_dim, 1)
        self.out_proj = nn.Linear(self.latent_dim, self.emb_dim)

    def from_codes(self, codes: torch.Tensor, codec):
        """
        get a sequence of continuous embeddings from a sequence of discrete codes.
        unlike it's counterpart in the original VQ-VAE, this function adds for any special tokens
        necessary for the language model, like <MASK>.
        """
        n_codebooks = codes.shape[1]
        latent = []
        for i in range(n_codebooks):
            c = codes[:, i, :]
            lookup_table = codec[i]#.weight
            if hasattr(self, "special"):
                special_lookup = torch.cat(
                    [self.special[tkn][i : i + 1] for tkn in self.special], dim=0
                )
                lookup_table = torch.cat([lookup_table, special_lookup], dim=0)

            l = F.embedding(c, lookup_table)
            # print(i, l.shape)
            latent.append(l)
        # [(B, 128, T), ....]
        latent = sum(latent)
        # latent = torch.stack(latent, dim=-1)
        # B, 8, T, K
        return latent

    def forward(self, latents: torch.Tensor):
        """
        project a sequence of latents to a sequence of embeddings
        """
        x = self.out_proj(latents)
        # x = torch.stack([self.out_proj[k](latents[:, :, k]) for k in range(4)], dim=2)
        return x

def codebook_flatten(tokens: torch.Tensor):
    """
    flatten a sequence of tokens from (batch, codebook, time) to (batch, codebook * time)
    """
    return rearrange(tokens, "b c t -> b (t c)")

def codebook_unflatten(flat_tokens: torch.Tensor, n_c: int = None):
    """
    unflatten a sequence of tokens from (batch, codebook * time) to (batch, codebook, time)
    """
    tokens = rearrange(flat_tokens, "b (t c) -> b c t", c=n_c)
    return tokens

def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)

def variable_cosine_schedule(t, phi):
    return torch.cos((t - phi) * math.pi / 2)

def uniform(shape, min = 0, max = 1, device = None):
    return torch.zeros(shape, device = device).float().uniform_(0, 1)

def top_k(logits, k=256):
    val, ind = logits.topk(k, dim = -1)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(2, ind, val)
    return probs

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


class VT2AModel(nn.Module):
    def __init__(self,
                 audio_cb_path,
                 latent_dim=128,
                 n_codebooks=4,
                 num_tokens=2048,
                 audio_seq_len=500,
                 visual_seq_len=500,
                 dim=768,
                 num_layers=12,
                 num_heads=16,
                 dec_num_layers=12,
                 dec_num_heads=16,
                 weight_init='gaussian',
                 depthwise_init='current',
                 zero_bias_init=True,
                 mask_ratio_min=0.5,
                 mask_ratio_max=1.0,
                 mask_ratio_mu=0.55,
                 mask_ratio_std=0.25,
                 encoder_dim=4096,
                 use_eva_clip=False,
                 ct=False,
                 device="cuda",
                 ):
        super().__init__()
        self.mask_ratio_generator = stats.truncnorm((mask_ratio_min - mask_ratio_mu)/mask_ratio_std,
                                                    (mask_ratio_max - mask_ratio_mu)/mask_ratio_std,
                                                    loc=mask_ratio_mu, scale=mask_ratio_std)
        self.audio_seq_len = audio_seq_len
        self.visual_seq_len = visual_seq_len
        self.dim = dim
        self.quantizer = torch.load(audio_cb_path, map_location=device)
        self.modality_v = nn.Parameter(torch.zeros(1, 1, dim))
        self.modality_a = nn.Parameter(torch.zeros(1, 1, dim))
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,  self.visual_seq_len + self.audio_seq_len, dim), requires_grad=False)
        self.encoder = ContinuousTransformerWrapper(
            max_seq_len=self.audio_seq_len + visual_seq_len,
            use_abs_pos_emb=False,
            emb_dropout = 0.1,  # dropout after embedding
            attn_layers=EncoderKS(
                dim=dim,
                depth=num_layers,
                heads=num_heads,
                attn_flash=True,
                ff_no_bias=True,
                attn_dropout = 0.1,    # dropout post-attention
                ff_dropout = 0.1,       # feedforward dropout
                multiway=True,
                custom_layers=('a', 'ff')*num_layers,
                attn_alibi=True,
                alibi_pos_bias=True
            )
        )

        # self.decoder_pos_embed_learned  = nn.Parameter(torch.zeros(1, self.visual_seq_len + self.audio_seq_len, dim))
        self.decoder = ContinuousTransformerWrapper(
            max_seq_len=self.visual_seq_len + self.audio_seq_len,
            use_abs_pos_emb=False,
            emb_dropout = 0.1,  # dropout after embedding
            attn_layers=EncoderKS(
                dim=dim,
                depth=dec_num_layers,
                heads=dec_num_heads,
                attn_flash=True,
                ff_no_bias=True,
                attn_dropout = 0.1,    # dropout post-attention
                ff_dropout = 0.1,       # feedforward dropout
                # modulation=True,
                attn_alibi=True,
                alibi_pos_bias=True
            )
        )
        self.embedding = CodebookEmbedding(
            latent_dim=latent_dim,
            n_codebooks=n_codebooks,
            vocab_size=num_tokens,
            emb_dim=dim,
            special_tokens=["MASK"],
        )
        self.mask_token = self.embedding.special_idxs["MASK"]
        # Add final conv layer
        self.n_codebooks = n_codebooks
        self.classifier = nn.ModuleList([nn.Linear(dim, num_tokens, bias=False) for _ in range(n_codebooks)])
        # self.critic_head = nn.ModuleList([nn.Linear(dim, 1) for _ in range(n_codebooks)])
        if not use_eva_clip:
            self.img_linear = nn.Linear(encoder_dim, dim)
        elif dim != 768:
            self.img_linear = nn.Linear(768, dim)
        else:
            self.img_linear = nn.Identity()
        self.ct = ct
        if ct:
            self.ct_v_linear = nn.Linear(dim, 768)
            self.ct_a_linear = nn.Linear(dim, 768)
        else:
            self.ct_v_linear = nn.Identity()
            self.ct_a_linear = nn.Identity()
        self._init_weights(weight_init, depthwise_init, zero_bias_init)

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (str, optional): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (str, optional): Depthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initialize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for layer_idx, tr_layer in enumerate(self.encoder.attn_layers.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.encoder.attn_layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for layer_idx, tr_layer in enumerate(self.decoder.attn_layers.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.decoder.attn_layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        pos_embed = np_get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.audio_seq_len + self.visual_seq_len, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)
        # torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)

    # @torch.no_grad()
    def audio_only(self, a):
        full_a = self.embedding.from_codes(a, self.quantizer)
        full_a = self.embedding(full_a)
        full_a = rearrange(full_a, "b d t -> b t d")
        full_a = full_a + self.pos_embed[:, self.visual_seq_len:] + self.modality_a
        a_embs = self.encoder(full_a, split_pos=0)
        return a_embs.mean(dim=1)
    
    def visual_only(self, v):
        v = self.img_linear(v)
        v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
        v_embs = self.encoder(v, split_pos=-1)
        return v_embs.mean(dim=1)
    
    def audio_visual(self, a, v):
        v = self.img_linear(v)
        if isinstance(a, list):
            a = self.embedding.from_codes(a[0], self.quantizer)*a[2] + self.embedding.from_codes(a[1], self.quantizer)*(1 - a[2])
        else:
            a = self.embedding.from_codes(a, self.quantizer)
        a = self.embedding(a)
        a = rearrange(a, "b d t -> b t d")
        v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
        a = a + self.pos_embed[:, v.shape[1]:] + self.modality_a
        av_embs = self.encoder(torch.cat([v, a], dim=1), split_pos=v.shape[1], av_order=['v', 'a'])
        # av_embs = self.decoder(av_embs)
        return torch.mean(av_embs, dim=1)
    
    def contrastive_forward(self, a, v):
        v = self.img_linear(v)
        a = self.embedding.from_codes(a, self.quantizer)
        a = self.embedding(a)
        a = rearrange(a, "b d t -> b t d")
        v_embs = self.encoder(v + self.pos_embed[:, :v.shape[1]] + self.modality_v, split_pos=-1)
        a_embs = self.encoder(a + self.pos_embed[:, v.shape[1]:] + self.modality_a, split_pos=0)
        v_embs = F.normalize(self.ct_v_linear(v_embs.mean(dim=1)), dim=-1)
        a_embs = F.normalize(self.ct_a_linear(a_embs.mean(dim=1)), dim=-1)
        return a_embs, v_embs
    
    def forward(self, a, v, pad_mask, return_embs=False, return_logits=True): # phi mean  phi, eval_flag,
        # print('unflatten a')
        # print(a)
        # flat_a = codebook_flatten(a)
        # print('flat a')
        # print(flat_a.shape)
        # print(flat_a)
        
        v = self.img_linear(v)
        B, K, T = a.shape
        # get random probability
        # if not eval_flag:
            # mask_ratio_std = 0.25
            # mask_ratio_min = 0.0
            # mask_ratio_max = 1.0
            # mask_ratio_generator = stats.truncnorm((mask_ratio_min - mean)/mask_ratio_std,
            #                                             (mask_ratio_max - mean)/mask_ratio_std,
            #                                             loc=mean, scale=mask_ratio_std)
        r = self.mask_ratio_generator.rvs(1)[0]
        mask = random_b(a, r)

            # t = uniform((B, ), device=v.device)
            # r = variable_cosine_schedule(t, phi)
            # mask = random_b2(a, r)
        
        # mask = self.forward_confidence_to_get_mask(a, v, pad_mask, r)
        # self.train()
            
        # else:
        #     t = uniform((B, ), device=v.device)
        #     # r = variable_cosine_schedule(t, phi)
        #     r = cosine_schedule(t)
        #     mask = random_b2(a, r)

        # mask = random_b2(a, t)
        # print(mask)
        # mask = codebook_unmask(mask, self.n_conditioning_codebooks)
        # print(mask)
        masked_a, mask = apply_mask(a, mask, mask_token=self.mask_token)
        target = codebook_flatten(a)
        flat_mask = codebook_flatten(mask)
        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), -1)
        masked_a = self.embedding.from_codes(masked_a, self.quantizer)
        # print(masked_a.shape)
        # B, 8, T, K
        # masked_a = rearrange(masked_a, 'b d t k -> b t k d')
        masked_a = self.embedding(masked_a)
        # masked_a = rearrange(masked_a, "b d t -> b t d")
        # print("HHH", masked_a.shape, self.pos_embed[:, v.shape[1]:].shape)
        # masked_a = rearrange(masked_a, 'b t k d -> b (t k) d')
        # v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
        # print(v.shape)
        masked_a = masked_a + self.pos_embed[:, v.shape[1]:][:, :masked_a.shape[1]] + self.modality_a
        if return_embs:
            v_embs = self.encoder(v + self.pos_embed[:, :v.shape[1]] + self.modality_v, split_pos=-1)
            a_embs = self.encoder(masked_a, split_pos=0)
            if not return_logits:
                v_embs = F.normalize(self.ct_v_linear(v_embs.mean(dim=1)), dim=-1)
                a_embs = F.normalize(self.ct_a_linear(a_embs.mean(dim=1)), dim=-1)
                return a_embs, v_embs
        prob = random.random()
        if prob < 0.1:
            v = torch.zeros_like(v) + self.pos_embed[:, :v.shape[1]] + self.modality_v
        else:
            v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
        extend_mask = torch.cat([pad_mask, torch.ones(masked_a.shape[0], masked_a.shape[1]).to(masked_a.device)], dim=1)
        extend_mask = (extend_mask > 0)
        latents = self.encoder(torch.cat([v, masked_a], dim=1), split_pos=v.shape[1], av_order=['v', 'a'], mask=extend_mask)
        
        # v_cond = torch.repeat_interleave(latents[:, :v.shape[1], :], 50, dim=1)
        # B, K, T, D
        # dec_input = latents + self.decoder_pos_embed_learned
        # out: b, t, d
        out = self.decoder(latents, mask=extend_mask)
        # logits: b, k, t, d
        logits = torch.stack([self.classifier[k](out[:, v.shape[1]:, :]) for k in range(K)], dim=1)
        logits = rearrange(logits, "b k t d -> b d (k t)")
        return logits, t_masked        
        # sampled_ids = gumbel_sample_v1(logits, temperature = 1.0)
        # logits = rearrange(logits, "b k t d -> b d (k t)")
        # # print(mask.shape, sampled_ids.shape)
        # generated = torch.where(mask.bool(), sampled_ids, a) # 22, 4, 500
        # generated = self.embedding.from_codes(generated, self.quantizer)
        # generated = self.embedding(generated)
        # generated = generated + self.pos_embed[:, v.shape[1]:][:, :generated.shape[1]] + self.modality_a
        # latents_2 = self.encoder(torch.cat([v, generated], dim=1), split_pos=v.shape[1], av_order=['v', 'a'], mask=extend_mask)
        # out_2 = self.decoder(latents_2, mask=extend_mask)
        # critic_logits = torch.stack([self.critic_head[k](out_2[:, v.shape[1]:, :]) for k in range(K)], dim=1)
        # critic_logits = critic_logits.squeeze(-1)#rearrange(critic_logits, "b k t d -> b d (k t)")
        # critic_labels = (sampled_ids != a).float()



        # return logits, t_masked, critic_logits, critic_labels
    
    @torch.no_grad()
    def forward_confidence_to_get_mask(self, a, v, pad_mask, mask_ratio):
        self.eval()
        B, K, T = a.shape
        mask = torch.ones_like(a).to(v.device).long()
        masked_a, mask = apply_mask(a, mask, mask_token=self.mask_token)
        target = codebook_flatten(a)
        flat_mask = codebook_flatten(mask)
        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), -1)
        masked_a = self.embedding.from_codes(masked_a, self.quantizer)
        masked_a = self.embedding(masked_a)
        masked_a = masked_a + self.pos_embed[:, v.shape[1]:][:, :masked_a.shape[1]] + self.modality_a
        if random.random() < 0.1:
            v = torch.zeros_like(v) + self.pos_embed[:, :v.shape[1]] + self.modality_v
        else:
            v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
        extend_mask = torch.cat([pad_mask, torch.ones(masked_a.shape[0], masked_a.shape[1]).to(masked_a.device)], dim=1)
        extend_mask = (extend_mask > 0)
        latents = self.encoder(torch.cat([v, masked_a], dim=1), split_pos=v.shape[1], av_order=['v', 'a'], mask=extend_mask)
        out = self.decoder(latents, mask=extend_mask)
        # logits: b, k, t, d
        logits = torch.stack([self.classifier[k](out[:, v.shape[1]:, :]) for k in range(K)], dim=1)
        logits = rearrange(logits, "b k t d -> b (k t) d")
        sampled_a, selected_probs = sample_from_logits(
                logits, sample=True,
                temperature=1.0,
                typical_filtering=False,
                typical_mass=0.1,
                typical_min_tokens=1,
                top_k=256,
                top_p=None,
                return_probs=True,
            )
        num_to_mask = torch.tensor([K * T * mask_ratio]).unsqueeze(1).long().to(v.device)
        mask_temperature = 2.0
        mask = mask_by_random_topk(
                num_to_mask, selected_probs, mask_temperature * torch.tensor(1.0).to(v.device)
            ).bool().long()
        mask = codebook_unflatten(mask, K)
        return mask

    @torch.no_grad()
    def generate_audio_sample(self, v, pad_mask,
                              sampling_temperature=1.,
                              sampling_steps=16, #36,
                              mask_temperature: float = 10.5,
                              typical_filtering=False,
                              typical_mass=0.2,
                              typical_min_tokens=1,
                              top_p=None,
                              sample_cutoff: float = 1.0,
                              cfg_coef = 3.0,
                              ):
        # b k t
        shape = (v.shape[0], 4, 500)
        v = self.img_linear(v)
        null_v = torch.zeros_like(v).to(v.device) + self.pos_embed[:, :v.shape[1]] + self.modality_v
        v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
        
        a = torch.full(shape, self.mask_token, dtype=torch.long).to(v.device)
        mask = torch.ones_like(a).to(v.device).int()
        # apply the mask to z
        masked_a = a.masked_fill(mask.bool(), self.mask_token)

        # how many mask tokens to begin with?
        num_mask_tokens_at_start = (masked_a == self.mask_token).sum()
        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks

        for i in range(sampling_steps):
            r = scalar_to_batch_tensor((i + 1) / sampling_steps, a.shape[0]).to(a.device)
            # get latents
            audio_emb_tokens = self.embedding.from_codes(masked_a, self.quantizer)
            audio_emb_tokens = self.embedding(audio_emb_tokens)
            # audio_emb_tokens = rearrange(audio_emb_tokens, "b d t -> b t d")
            audio_emb_tokens = audio_emb_tokens + self.pos_embed[:, v.shape[1]:][:, :audio_emb_tokens.shape[1]] + self.modality_a
            enc_input = torch.cat([v, audio_emb_tokens], dim=1)
            uncond_enc_input = torch.cat([null_v, audio_emb_tokens], dim=1)

            extend_mask = torch.cat([pad_mask, torch.ones(audio_emb_tokens.shape[0], audio_emb_tokens.shape[1]).to(audio_emb_tokens.device)], dim=1)
            extend_mask = (extend_mask > 0)

            dec_input = self.encoder(torch.cat([enc_input, uncond_enc_input], dim=0), split_pos=v.shape[1], av_order=['v', 'a'], mask=extend_mask)
            # dec_input = self.encoder(torch.cat([v, audio_emb_tokens], dim=1), split_pos=v.shape[1], av_order=['v', 'a'])
            # uncond_dec_input = self.encoder(torch.cat([null_v, audio_emb_tokens], dim=1), split_pos=v.shape[1], av_order=['v', 'a'])
            # B, K, T, D
            # dec_input = latents + self.decoder_pos_embed_learned
            # out: b, t, d
            out = self.decoder(dec_input, mask=extend_mask)
            # uncond_out = self.decoder(uncond_dec_input)
            # logits: b, k, t, d
            audio_logits = torch.stack([self.classifier[k](out[:1, v.shape[1]:]) for k in range(4)], dim=1)
            uncond_audio_logits = torch.stack([self.classifier[k](out[1:, v.shape[1]:]) for k in range(4)], dim=1)
            audio_logits = uncond_audio_logits + (audio_logits - uncond_audio_logits) * cfg_coef
            
            # b (d k) t -> b d (t k) -> b (t k) d
            audio_logits = rearrange(audio_logits, "b k t d -> b (k t) d")
            b = audio_logits.shape[0]
            # b (k t) d
            sampled_a, selected_probs = sample_from_logits(
                audio_logits, sample=(
                        (i / sampling_steps) <= sample_cutoff
                ),
                temperature=sampling_temperature,
                typical_filtering=typical_filtering,
                typical_mass=typical_mass,
                typical_min_tokens=typical_min_tokens,
                top_k=256,
                top_p=top_p,
                return_probs=True,
            )
            # flatten z_masked and mask, so we can deal with the sampling logic
            # we'll unflatten them at the end of the loop for the next forward pass
            # remove conditioning codebooks, we'll add them back at the end
            # b k t -> b (t k)
            masked_a = codebook_flatten(masked_a)

            mask = (masked_a == self.mask_token).int()
            # print(masked_a.shape, sampled_a.shape, mask.shape)
            sampled_a = torch.where(mask.bool(), sampled_a, masked_a)
            selected_probs = torch.where(
                mask.bool(), selected_probs, torch.inf
            )
            num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()
            if i != (sampling_steps - 1):
                num_to_mask = torch.maximum(
                    torch.tensor(1),
                    torch.minimum(
                        mask.sum(dim=-1, keepdim=True) - 1,
                        num_to_mask
                    )
                )
            # new mask
            mask = mask_by_random_topk(
                num_to_mask, selected_probs, mask_temperature * (1 - r)
            )
            # update mask
            masked_a = torch.where(
                mask.bool(), self.mask_token, sampled_a
            )
            masked_a = codebook_unflatten(masked_a, n_infer_codebooks)
            mask = codebook_unflatten(mask, n_infer_codebooks)
            # add conditioning codebooks back to z_masked
            # masked_a = torch.cat(
            #     (a[:, :self.n_conditioning_codebooks, :], masked_a), dim=1
            # )
        # add conditioning codebooks back to sampled_z
        sampled_a = codebook_unflatten(sampled_a, n_infer_codebooks)
        # sampled_a = torch.cat(
        #     (a[:, :self.n_conditioning_codebooks, :], sampled_a), dim=1
        # )
        return sampled_a
    

    @torch.no_grad()
    def generate_audio_sample_cond_text(self, v, v_t, pad_mask,
                              sampling_temperature=1.,
                              sampling_steps=16, #36,
                              mask_temperature: float = 10.5,
                              typical_filtering=False,
                              typical_mass=0.2,
                              typical_min_tokens=1,
                              top_p=None,
                              sample_cutoff: float = 1.0,
                              cfg_coef = 3.0,
                              ):
        # b k t
        shape = (v.shape[0], 4, 500)
        v = self.img_linear(v)
        # v_t = self.img_linear(v_t)
        null_v = torch.zeros_like(v).to(v.device) + self.pos_embed[:, :v.shape[1]] + self.modality_v
        # null_vt = torch.zeros_like(v_t).to(v_t.device) + self.pos_embed[:, :v_t.shape[1]] + self.modality_v
        
        v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v

        # v_t = v_t + self.pos_embed[:, :v_t.shape[1]] + self.modality_v
        
        a = torch.full(shape, self.mask_token, dtype=torch.long).to(v.device)
        mask = torch.ones_like(a).to(v.device).int()
        # apply the mask to z
        masked_a = a.masked_fill(mask.bool(), self.mask_token)

        # how many mask tokens to begin with?
        num_mask_tokens_at_start = (masked_a == self.mask_token).sum()
        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks

        for i in range(sampling_steps):
            r = scalar_to_batch_tensor((i + 1) / sampling_steps, a.shape[0]).to(a.device)
            # get latents
            audio_emb_tokens = self.embedding.from_codes(masked_a, self.quantizer)
            audio_emb_tokens = self.embedding(audio_emb_tokens)
            # audio_emb_tokens = rearrange(audio_emb_tokens, "b d t -> b t d")
            # audio_emb_tokens_t = audio_emb_tokens + self.pos_embed[:, v_t.shape[1]:][:, :audio_emb_tokens.shape[1]] + self.modality_a
            audio_emb_tokens_v = audio_emb_tokens + self.pos_embed[:, v.shape[1]:][:, :audio_emb_tokens.shape[1]] + self.modality_a
            # enc_input = torch.cat([v_t, audio_emb_tokens_t], dim=1)
            # uncond_enc_input_t = torch.cat([null_vt, audio_emb_tokens_t], dim=1)
            uncond_enc_input_v = torch.cat([null_v, audio_emb_tokens_v], dim=1)
            enc_input_v = torch.cat([v, audio_emb_tokens_v], dim=1)

            # extend_mask = torch.cat([pad_mask, torch.ones(audio_emb_tokens_t.shape[0], audio_emb_tokens_t.shape[1]).to(audio_emb_tokens_t.device)], dim=1)
            # extend_mask = (extend_mask > 0)

            # dec_input = self.encoder(torch.cat([enc_input, uncond_enc_input_t], dim=0), split_pos=v_t.shape[1], av_order=['v', 'a'])
            
            dec_input_v = self.encoder(torch.cat([enc_input_v, uncond_enc_input_v], dim=0), split_pos=v.shape[1], av_order=['v', 'a'])
            
            # dec_input = self.encoder(torch.cat([v, audio_emb_tokens], dim=1), split_pos=v.shape[1], av_order=['v', 'a'])
            # uncond_dec_input = self.encoder(torch.cat([null_v, audio_emb_tokens], dim=1), split_pos=v.shape[1], av_order=['v', 'a'])
            # B, K, T, D
            # dec_input = latents + self.decoder_pos_embed_learned
            # out: b, t, d
            # out = self.decoder(dec_input)
            out_v = self.decoder(dec_input_v)
            # uncond_out = self.decoder(uncond_dec_input)
            # logits: b, k, t, d
            # audio_logits_t = torch.stack([self.classifier[k](out[:1, v_t.shape[1]:]) for k in range(4)], dim=1)
            # uncond_audio_logits_t = torch.stack([self.classifier[k](out[1:, v_t.shape[1]:]) for k in range(4)], dim=1)
            audio_logits_v = torch.stack([self.classifier[k](out_v[:1, v.shape[1]:]) for k in range(4)], dim=1)
            uncond_audio_logits_v = torch.stack([self.classifier[k](out_v[1:, v.shape[1]:]) for k in range(4)], dim=1)

            # audio_logits_v_gen = uncond_audio_logits + (audio_logits_v - uncond_audio_logits) * cfg_coef
            # audio_logits_t_gen = uncond_audio_logits + (audio_logits_t - uncond_audio_logits) * cfg_coef
            # audio_logits = audio_logits_v_gen + (audio_logits_t_gen - audio_logits_v_gen) * 3.0
            audio_logits = uncond_audio_logits_v + (audio_logits_v - uncond_audio_logits_v) * 4.5 #(audio_logits_t - audio_logits_v) * 1.0 +


            # b (d k) t -> b d (t k) -> b (t k) d
            audio_logits = rearrange(audio_logits, "b k t d -> b (k t) d")
            b = audio_logits.shape[0]
            # b (k t) d
            sampled_a, selected_probs = sample_from_logits(
                audio_logits, sample=(
                        (i / sampling_steps) <= sample_cutoff
                ),
                temperature=1.0,
                typical_filtering=typical_filtering,
                typical_mass=typical_mass,
                typical_min_tokens=typical_min_tokens,
                top_k=256,
                top_p=top_p,
                return_probs=True,
            )
            # flatten z_masked and mask, so we can deal with the sampling logic
            # we'll unflatten them at the end of the loop for the next forward pass
            # remove conditioning codebooks, we'll add them back at the end
            # b k t -> b (t k)
            masked_a = codebook_flatten(masked_a)

            mask = (masked_a == self.mask_token).int()
            # print(masked_a.shape, sampled_a.shape, mask.shape)
            sampled_a = torch.where(mask.bool(), sampled_a, masked_a)
            selected_probs = torch.where(
                mask.bool(), selected_probs, torch.inf
            )
            num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()
            if i != (sampling_steps - 1):
                num_to_mask = torch.maximum(
                    torch.tensor(1),
                    torch.minimum(
                        mask.sum(dim=-1, keepdim=True) - 1,
                        num_to_mask
                    )
                )
            # new mask
            mask = mask_by_random_topk(
                num_to_mask, selected_probs, mask_temperature * (1 - r)
            )
            # update mask
            masked_a = torch.where(
                mask.bool(), self.mask_token, sampled_a
            )
            masked_a = codebook_unflatten(masked_a, n_infer_codebooks)
            mask = codebook_unflatten(mask, n_infer_codebooks)
            # add conditioning codebooks back to z_masked
            # masked_a = torch.cat(
            #     (a[:, :self.n_conditioning_codebooks, :], masked_a), dim=1
            # )
        # add conditioning codebooks back to sampled_z
        sampled_a = codebook_unflatten(sampled_a, n_infer_codebooks)
        # sampled_a = torch.cat(
        #     (a[:, :self.n_conditioning_codebooks, :], sampled_a), dim=1
        # )
        return sampled_a
    

    @torch.no_grad()
    def generate_audio_sample_critic(self, v, pad_mask,
                              sampling_temperature=1.,
                              sampling_steps=16, #36,
                              mask_temperature: float = 10.5,
                              typical_filtering=False,
                              typical_mass=0.2,
                              typical_min_tokens=1,
                              top_p=None,
                              sample_cutoff: float = 1.0,
                              cfg_coef = 3.0,
                              ):
        # b k t
        shape = (v.shape[0], 4, 500)
        v = self.img_linear(v)
        null_v = torch.zeros_like(v).to(v.device) + self.pos_embed[:, :v.shape[1]] + self.modality_v
        v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
        
        a = torch.full(shape, self.mask_token, dtype=torch.long).to(v.device)
        mask = torch.ones_like(a).to(v.device).int()
        # apply the mask to z
        masked_a = a.masked_fill(mask.bool(), self.mask_token)

        # how many mask tokens to begin with?
        num_mask_tokens_at_start = (masked_a == self.mask_token).sum()
        # how many codebooks are we inferring vs conditioning on?
        n_infer_codebooks = self.n_codebooks

        for i in range(sampling_steps):
            r = scalar_to_batch_tensor((i + 1) / sampling_steps, a.shape[0]).to(a.device)
            # get latents
            audio_emb_tokens = self.embedding.from_codes(masked_a, self.quantizer)
            audio_emb_tokens = self.embedding(audio_emb_tokens)
            # audio_emb_tokens = rearrange(audio_emb_tokens, "b d t -> b t d")
            audio_emb_tokens = audio_emb_tokens + self.pos_embed[:, v.shape[1]:][:, :audio_emb_tokens.shape[1]] + self.modality_a
            enc_input = torch.cat([v, audio_emb_tokens], dim=1)
            uncond_enc_input = torch.cat([null_v, audio_emb_tokens], dim=1)

            extend_mask = torch.cat([pad_mask, torch.ones(audio_emb_tokens.shape[0], audio_emb_tokens.shape[1]).to(audio_emb_tokens.device)], dim=1)
            extend_mask = (extend_mask > 0)

            dec_input = self.encoder(torch.cat([enc_input, uncond_enc_input], dim=0), split_pos=v.shape[1], av_order=['v', 'a'])
            out = self.decoder(dec_input)
            # logits: b, k, t, d
            audio_logits = torch.stack([self.classifier[k](out[:1, v.shape[1]:]) for k in range(4)], dim=1)
            uncond_audio_logits = torch.stack([self.classifier[k](out[1:, v.shape[1]:]) for k in range(4)], dim=1)
            audio_logits = uncond_audio_logits + (audio_logits - uncond_audio_logits) * cfg_coef
            
            # b (d k) t -> b d (t k) -> b (t k) d
            audio_logits = rearrange(audio_logits, "b k t d -> b (k t) d")

            b = audio_logits.shape[0]
            # b (k t) d
            sampled_a, selected_probs = sample_from_logits(
                audio_logits, sample=(
                        (i / sampling_steps) <= sample_cutoff
                ),
                temperature=sampling_temperature,
                typical_filtering=typical_filtering,
                typical_mass=typical_mass,
                typical_min_tokens=typical_min_tokens,
                top_k=256,
                top_p=top_p,
                return_probs=True,
            )
            masked_a = codebook_flatten(masked_a)
            mask = (masked_a == self.mask_token).int()
            generated_raw = torch.where(mask.bool(), sampled_a, masked_a)
            generated = codebook_unflatten(generated_raw, n_infer_codebooks)
            generated = self.embedding.from_codes(generated, self.quantizer)
            generated = self.embedding(generated)
            generated = generated + self.pos_embed[:, v.shape[1]:][:, :generated.shape[1]] + self.modality_a
            latents_2 = self.encoder(torch.cat([v, generated], dim=1), split_pos=v.shape[1], av_order=['v', 'a'], mask=extend_mask)
            out_2 = self.decoder(latents_2, mask=extend_mask)
            critic_logits = torch.stack([self.critic_head[k](out_2[:, v.shape[1]:, :]) for k in range(4)], dim=1)
            critic_logits = critic_logits.squeeze(-1)
            critic_probs = torch.sigmoid(critic_logits)
            # print(i, critic_probs[0], selected_probs[0])
            critic_logits = codebook_flatten(critic_logits)
            # flatten z_masked and mask, so we can deal with the sampling logic
            # we'll unflatten them at the end of the loop for the next forward pass
            # remove conditioning codebooks, we'll add them back at the end
            # b k t -> b (t k)
            # masked_a = codebook_flatten(masked_a)

            # mask = (masked_a == self.mask_token).int()
            # print(masked_a.shape, sampled_a.shape, mask.shape)
            # sampled_a = torch.where(mask.bool(), sampled_a, masked_a)
            # selected_probs = torch.where(
            #     mask.bool(), selected_probs, torch.inf
            # )
            num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long()
            if i != (sampling_steps - 1):
                num_to_mask = torch.maximum(
                    torch.tensor(1),
                    torch.minimum(
                        mask.sum(dim=-1, keepdim=True) - 1,
                        num_to_mask
                    )
                )

            noise = gumbel_noise_like(critic_logits)
            confidence = - (critic_logits + mask_temperature * (1 - r) * noise)
            sorted_confidence, sorted_idx = confidence.sort(dim=-1)
            # get the cut off threshold, given the mask length
            cut_off = torch.take_along_dim(
                sorted_confidence, num_to_mask, axis=-1
            )
            # mask out the tokens
            mask = confidence < cut_off

            # update mask
            # print("HHH", mask.shape, generated_raw.shape)
            masked_a = torch.where(
                mask.bool(), self.mask_token, generated_raw
            )
            masked_a = codebook_unflatten(masked_a, n_infer_codebooks)
            mask = codebook_unflatten(mask, n_infer_codebooks)
            # add conditioning codebooks back to z_masked
            # masked_a = torch.cat(
            #     (a[:, :self.n_conditioning_codebooks, :], masked_a), dim=1
            # )
        # add conditioning codebooks back to sampled_z
        generated_raw = codebook_unflatten(generated_raw, n_infer_codebooks)
        # sampled_a = torch.cat(
        #     (a[:, :self.n_conditioning_codebooks, :], sampled_a), dim=1
        # )
        return generated_raw
    
    @torch.no_grad()  
    def generate_audio_sample_refine(self, v, pad_mask,
                                sampling_temperature=1.,
                                sampling_steps=16, #36,
                                mask_temperature: float = 10.5,
                                typical_filtering=False,
                                typical_mass=0.2,
                                typical_min_tokens=1,
                                top_p=None,
                                sample_cutoff: float = 1.0,
                                cfg_coef = 3.0,):
            mask_temperature = 3.5
            thres = int(sampling_steps * 0.5) + 1
            # b k t
            shape = (v.shape[0], 4, 500)
            v = self.img_linear(v)
            null_v = torch.zeros_like(v).to(v.device) + self.pos_embed[:, :v.shape[1]] + self.modality_v
            v = v + self.pos_embed[:, :v.shape[1]] + self.modality_v
            
            a = torch.full(shape, self.mask_token, dtype=torch.long).to(v.device)
            mask = torch.ones_like(a).to(v.device).int()
            # apply the mask to z
            masked_a = a.masked_fill(mask.bool(), self.mask_token)

            # how many mask tokens to begin with?
            num_mask_tokens_at_start = (masked_a == self.mask_token).sum()
            # how many codebooks are we inferring vs conditioning on?
            n_infer_codebooks = self.n_codebooks
            mask_before_half = None
            for i in range(sampling_steps + (sampling_steps - thres)):
                if i < sampling_steps:
                    r = scalar_to_batch_tensor((i + 1) / sampling_steps, a.shape[0]).to(a.device)
                else:
                    r = scalar_to_batch_tensor((i - sampling_steps + thres) / sampling_steps, a.shape[0]).to(a.device)
                # get latents
                audio_emb_tokens = self.embedding.from_codes(masked_a, self.quantizer)
                audio_emb_tokens = self.embedding(audio_emb_tokens)
                # audio_emb_tokens = rearrange(audio_emb_tokens, "b d t -> b t d")
                audio_emb_tokens = audio_emb_tokens + self.pos_embed[:, v.shape[1]:][:, :audio_emb_tokens.shape[1]] + self.modality_a
                enc_input = torch.cat([v, audio_emb_tokens], dim=1)
                uncond_enc_input = torch.cat([null_v, audio_emb_tokens], dim=1)

                extend_mask = torch.cat([pad_mask, torch.ones(audio_emb_tokens.shape[0], audio_emb_tokens.shape[1]).to(audio_emb_tokens.device)], dim=1)
                extend_mask = (extend_mask > 0)

                dec_input = self.encoder(torch.cat([enc_input, uncond_enc_input], dim=0), split_pos=v.shape[1], av_order=['v', 'a'])
                # dec_input = self.encoder(torch.cat([v, audio_emb_tokens], dim=1), split_pos=v.shape[1], av_order=['v', 'a'])
                # uncond_dec_input = self.encoder(torch.cat([null_v, audio_emb_tokens], dim=1), split_pos=v.shape[1], av_order=['v', 'a'])
                # B, K, T, D
                # dec_input = latents + self.decoder_pos_embed_learned
                # out: b, t, d
                out = self.decoder(dec_input)
                # uncond_out = self.decoder(uncond_dec_input)
                # logits: b, k, t, d
                audio_logits = torch.stack([self.classifier[k](out[:1, v.shape[1]:]) for k in range(4)], dim=1)
                uncond_audio_logits = torch.stack([self.classifier[k](out[1:, v.shape[1]:]) for k in range(4)], dim=1)
                audio_logits = uncond_audio_logits + (audio_logits - uncond_audio_logits) * cfg_coef
                
                # b (d k) t -> b d (t k) -> b (t k) d
                audio_logits = rearrange(audio_logits, "b k t d -> b (k t) d")
                b = audio_logits.shape[0]
                # audio_logits = top_k(audio_logits, k=256)
                # selected_probs, sampled_a = gumbel_sample(audio_logits, temperature=mask_temperature * _gamma(r))

                # b (k t) d
                if i < sampling_steps:
                    should_sample = (i / sampling_steps) <= sample_cutoff
                else:
                    should_sample = ((i - sampling_steps + thres) / sampling_steps) <= sample_cutoff
                sampled_a, selected_probs = sample_from_logits(
                    audio_logits, sample=should_sample,
                    temperature=mask_temperature * _gamma(r),
                    typical_filtering=typical_filtering,
                    typical_mass=typical_mass,
                    typical_min_tokens=typical_min_tokens,
                    top_k=256,
                    top_p=top_p,
                    return_probs=True,
                )
                # flatten z_masked and mask, so we can deal with the sampling logic
                # we'll unflatten them at the end of the loop for the next forward pass
                # remove conditioning codebooks, we'll add them back at the end
                # b k t -> b (t k)
                masked_a = codebook_flatten(masked_a)

                mask = (masked_a == self.mask_token).int()
                # print(masked_a.shape, sampled_a.shape, mask.shape)
                sampled_a = torch.where(mask.bool(), sampled_a, masked_a)
                selected_probs = torch.where(
                    mask.bool(), selected_probs, torch.inf
                )
                num_to_mask = torch.floor(_gamma(r) * num_mask_tokens_at_start).unsqueeze(1).long() #
                if i != (sampling_steps - 1) and i != (sampling_steps + (sampling_steps - thres) - 1):
                    num_to_mask = torch.maximum(
                        torch.tensor(1),
                        torch.minimum(
                            mask.sum(dim=-1, keepdim=True) - 1,
                            num_to_mask
                        )
                    )
                # new mask
                uncertainty = mask_temperature * _gamma(r) #* (1 - r) #_gamma_new(r) #(1 - r)
                mask = mask_by_random_topk(
                    num_to_mask, selected_probs, uncertainty #mask_temperature * (1 - r)
                )
                # mask = mask_by_topk(num_to_mask, selected_probs)
                if i == thres:
                    mask_before_half = mask.clone()
                # update mask
                masked_a = torch.where(
                    mask.bool(), self.mask_token, sampled_a
                )
                masked_a = codebook_unflatten(masked_a, n_infer_codebooks)
                mask = codebook_unflatten(mask, n_infer_codebooks)
                if i == sampling_steps - 1:
                    break
                    sampled_a = codebook_unflatten(sampled_a, n_infer_codebooks)
                    # mask_before_half = codebook_unflatten(mask_before_half, n_infer_codebooks)
                    mask_before_half = torch.ones_like(sampled_a).to(v.device).int()
                    mask_before_half[:,:2,:] = 0
                    masked_a = sampled_a.masked_fill((mask_before_half).bool(), self.mask_token)
                    # print(mask_before_half, torch.sum(mask_before_half), mask_before_half.shape)
                    # print(masked_a)
                    # print(self.mask_token)
                    # print("-" * 10)
                
                # add conditioning codebooks back to z_masked
                # masked_a = torch.cat(
                #     (a[:, :self.n_conditioning_codebooks, :], masked_a), dim=1
                # )
            sampled_a = codebook_unflatten(sampled_a, n_infer_codebooks)
            # add conditioning codebooks back to sampled_z
            
            # sampled_a = torch.cat(
            #     (a[:, :self.n_conditioning_codebooks, :], sampled_a), dim=1
            # )
            return sampled_a


def gumbel_noise_like(t):
    noise = torch.zeros_like(t).uniform_(1e-20, 1)
    return -torch.log(-torch.log(noise))


def mask_by_random_topk(
        num_to_mask: int,
        probs: torch.Tensor,
        temperature: torch.Tensor,
):
    """
    Args:
        num_to_mask (int): number of tokens to mask
        probs (torch.Tensor): probabilities for each sampled event, shape (batch, seq)
        temperature (float, optional): temperature. Defaults to 1.0.
    """
    # logging.debug(f"masking by random topk")
    # logging.debug(f"num to mask: {num_to_mask}")
    # logging.debug(f"probs shape: {probs.shape}")
    # logging.debug(f"temperature: {temperature}")
    # logging.debug("")

    noise = gumbel_noise_like(probs)
    confidence = torch.log(probs) + temperature.unsqueeze(-1) * noise
    # logging.debug(f"confidence shape: {confidence.shape}")

    sorted_confidence, sorted_idx = confidence.sort(dim=-1)
    # logging.debug(f"sorted confidence shape: {sorted_confidence.shape}")
    # logging.debug(f"sorted idx shape: {sorted_idx.shape}")

    # get the cut off threshold, given the mask length
    cut_off = torch.take_along_dim(
        sorted_confidence, num_to_mask, axis=-1
    )
    # logging.debug(f"cut off shape: {cut_off.shape}")

    # mask out the tokens
    mask = confidence < cut_off
    # logging.debug(f"mask shape: {mask.shape}")

    return mask


def mask_by_topk(
        num_to_mask: int,
        logits: torch.Tensor,
):
    """
    Args:
        num_to_mask (int): number of tokens to mask
        logits (torch.Tensor): log probabilities for each sampled event, shape (batch, seq)
        temperature (float, optional): temperature. Defaults to 1.0.
    """
    # logging.debug(f"masking by random topk")
    # logging.debug(f"num to mask: {num_to_mask}")
    # logging.debug(f"probs shape: {probs.shape}")
    # logging.debug(f"temperature: {temperature}")
    # logging.debug("")

    confidence = logits
    # logging.debug(f"confidence shape: {confidence.shape}")

    sorted_confidence, sorted_idx = confidence.sort(dim=-1)
    # logging.debug(f"sorted confidence shape: {sorted_confidence.shape}")
    # logging.debug(f"sorted idx shape: {sorted_idx.shape}")

    # get the cut off threshold, given the mask length
    cut_off = torch.take_along_dim(
        sorted_confidence, num_to_mask, axis=-1
    )
    # logging.debug(f"cut off shape: {cut_off.shape}")

    # mask out the tokens
    mask = confidence < cut_off
    # logging.debug(f"mask shape: {mask.shape}")

    return mask


def sample_from_logits(
        logits,
        sample: bool = True,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        typical_filtering: bool = False,
        typical_mass: float = 0.2,
        typical_min_tokens: int = 1,
        return_probs: bool = False
):
    """Convenience function to sample from a categorial distribution with input as
    unnormalized logits.

    Parameters
    ----------
    logits : Tensor[..., vocab_size]
    config: SamplingConfig
        The set of hyperparameters to be used for sampling
        sample : bool, optional
            Whether to perform multinomial sampling, by default True
        temperature : float, optional
            Scaling parameter when multinomial samping, by default 1.0
        top_k : int, optional
            Restricts sampling to only `top_k` values acc. to probability,
            by default None
        top_p : float, optional
            Restricts sampling to only those values with cumulative
            probability = `top_p`, by default None

    Returns
    -------
    Tensor[...]
        Sampled tokens
    """
    shp = logits.shape[:-1]

    if typical_filtering:
        typical_filter(logits,
                       typical_mass=typical_mass,
                       typical_min_tokens=typical_min_tokens
                       )

    # Apply top_k sampling
    if top_k is not None:
        v, _ = logits.topk(top_k)
        logits[logits < v[..., [-1]]] = -float("inf")

    # Apply top_p (nucleus) sampling
    if top_p is not None and top_p < 1.0:
        v, sorted_indices = logits.sort(descending=True)
        cumulative_probs = v.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        # Right shift indices_to_remove to keep 1st token over threshold
        sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, 0), value=False)[
                                   ..., :-1
                                   ]

        # Compute indices_to_remove in unsorted array
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )

        logits[indices_to_remove] = -float("inf")

    # Perform multinomial sampling after normalizing logits
    probs = (
        F.softmax(logits / temperature, dim=-1)
        if temperature > 0
        else logits.softmax(dim=-1)
    )
    token = (
        probs.view(-1, probs.size(-1)).multinomial(1).squeeze(1).view(*shp)
        if sample
        else logits.argmax(-1)
    )

    if return_probs:
        token_probs = probs.take_along_dim(token.unsqueeze(-1), dim=-1).squeeze(-1)
        return token, token_probs
    else:
        return token


def typical_filter(
        logits,
        typical_mass: float = 0.95,
        typical_min_tokens: int = 1, ):
    nb, nt, _ = logits.shape
    x_flat = rearrange(logits, "b t l -> (b t ) l")
    x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
    x_flat_norm_p = torch.exp(x_flat_norm)
    entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

    c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
    c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
    x_flat_cumsum = (
        x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)
    )

    last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
    sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(
        1, last_ind.view(-1, 1)
    )
    if typical_min_tokens > 1:
        sorted_indices_to_remove[..., :typical_min_tokens] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, x_flat_indices, sorted_indices_to_remove
    )
    x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
    logits = rearrange(x_flat, "(b t) l -> b t l", t=nt)
    return logits


# if __name__ == "__main__":
#     # model = VT2AModel(audio_cb_path="/pscratch/sd/x/xiuliu/meta_pretrain_vgg_encodec_embed.pt")
#     model = VT2AModel(audio_cb_path="/pscratch/sd/x/xiuliu/dac_quantizer.pth")
#     model.cuda()
#     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#     params = sum([np.prod(p.size()) for p in model_parameters])
#     print("number of trainable params: ", params)