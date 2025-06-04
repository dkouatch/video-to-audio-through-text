import math
from functools import partial
import typing as tp
import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.x_transformers.custom_transformers import ContinuousTransformerWrapper, EncoderXL
from einops import rearrange, repeat
from vt2a.modules.pos_embed import np_get_1d_sincos_pos_embed
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
    log_prob = torch.log_softmax(t, dim=dim)
    noisy_logits = log_prob / max(temperature, 1e-10) + noise_vector
    token_probs, tokens = torch.max(noisy_logits, dim=dim)
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
        self.pos_embed = nn.Parameter(torch.zeros(1,  self.visual_seq_len + self.audio_seq_len, dim), requires_grad=False)
        self.encoder = ContinuousTransformerWrapper(
            max_seq_len=self.audio_seq_len + visual_seq_len,
            use_abs_pos_emb=False,
            emb_dropout = 0.1,  # dropout after embedding
            attn_layers=EncoderXL(
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

        self.decoder = ContinuousTransformerWrapper(
            max_seq_len=self.visual_seq_len + self.audio_seq_len,
            use_abs_pos_emb=False,
            emb_dropout = 0.1,  # dropout after embedding
            attn_layers=EncoderXL(
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

        pos_embed = np_get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.audio_seq_len + self.visual_seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.modality_a, std=.02)
        torch.nn.init.normal_(self.modality_v, std=.02)

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
        return torch.mean(av_embs, dim=1)
    
    def forward(self, a, v, pad_mask, return_logits=True):
        v = self.img_linear(v)
        B, K, T = a.shape
        r = self.mask_ratio_generator.rvs(1)[0]
        mask = random_b(a, r)
        masked_a, mask = apply_mask(a, mask, mask_token=self.mask_token)
        target = codebook_flatten(a)
        flat_mask = codebook_flatten(mask)
        # replace target with ignore index for masked tokens
        t_masked = target.masked_fill(~flat_mask.bool(), -1)
        masked_a = self.embedding.from_codes(masked_a, self.quantizer)
        masked_a = self.embedding(masked_a)
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
        out = self.decoder(latents, mask=extend_mask)
        # logits: b, k, t, d
        logits = torch.stack([self.classifier[k](out[:, v.shape[1]:, :]) for k in range(K)], dim=1)
        logits = rearrange(logits, "b k t d -> b d (k t)")
        return logits, t_masked

    @torch.no_grad()
    def generate_audio_sample(self, v, pad_mask,
                              sampling_temperature=1.,
                              sampling_steps=16,
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

        # add conditioning codebooks back to sampled_z
        sampled_a = codebook_unflatten(sampled_a, n_infer_codebooks)
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