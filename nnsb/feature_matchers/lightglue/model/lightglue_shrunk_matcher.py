import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.onnx import symbolic_helper

torch.backends.cudnn.deterministic = True

FUSE_MULTI_HEAD_ATTENTION = False
CUSTOM_OP_NAME = "fabiosim::multi_head_attention"


def multi_head_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> torch.Tensor:
    b, n, d = q.shape
    head_dim = d // num_heads
    q, k, v = (
        t.reshape((b, n, num_heads, head_dim)).transpose(1, 2) for t in (q, k, v)
    )
    return F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape((b, n, d))


fused_multi_head_attention = None


def multi_head_attention_dispatch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> torch.Tensor:
    if FUSE_MULTI_HEAD_ATTENTION and fused_multi_head_attention is not None:
        return fused_multi_head_attention(q, k, v, num_heads)
    else:
        return multi_head_attention(q, k, v, num_heads)


@symbolic_helper.parse_args("v", "v", "v", "i")
def symbolic_multi_head_attention(g, q, k, v, num_heads_i):
    return g.op(
        "com.microsoft::MultiHeadAttention", q, k, v, num_heads_i=num_heads_i
    ).setType(q.type())


def use_fused_multi_head_attention():
    global FUSE_MULTI_HEAD_ATTENTION, fused_multi_head_attention  # noqa: PLW0603
    FUSE_MULTI_HEAD_ATTENTION = True
    fused_multi_head_attention = torch.library.custom_op(
        CUSTOM_OP_NAME, mutates_args=()
    )(multi_head_attention)
    torch.onnx.register_custom_op_symbolic(
        CUSTOM_OP_NAME, symbolic_multi_head_attention, 9
    )


def multi_head_attention_dispatch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int
) -> torch.Tensor:
    if FUSE_MULTI_HEAD_ATTENTION and fused_multi_head_attention is not None:
        return fused_multi_head_attention(q, k, v, num_heads)
    else:
        return multi_head_attention(q, k, v, num_heads)


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(
        self, M: int, descriptor_dim: int, num_heads: int, gamma: float = 1.0
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = descriptor_dim // num_heads
        self.Wr = nn.Linear(M, head_dim // 2, bias=False)
        self.gamma = gamma
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines])
        return (
            emb.repeat_interleave(2, dim=3).repeat(1, 1, 1, self.num_heads).unsqueeze(4)
        )


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )


class SelfBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv: torch.Tensor = self.Wqkv(x)
        qkv = qkv.reshape((b, n, self.embed_dim, 3))
        qk, v = qkv[..., :2], qkv[..., 2]
        qk = self.apply_cached_rotary_emb(encoding, qk)
        q, k = qk[..., 0], qk[..., 1]
        context = multi_head_attention_dispatch(q, k, v, self.num_heads)
        message = self.out_proj(context)
        return x + self.ffn(torch.concat([x, message], 2))

    def rotate_half(self, qk: torch.Tensor) -> torch.Tensor:
        b, n, _, _ = qk.shape
        qk = qk.reshape((b, n, self.num_heads, self.head_dim // 2, 2, 2))
        qk = torch.stack((-qk[..., 1, :], qk[..., 0, :]), dim=4)
        qk = qk.reshape((b, n, self.embed_dim, 2))
        return qk

    def apply_cached_rotary_emb(
        self, encoding: torch.Tensor, qk: torch.Tensor
    ) -> torch.Tensor:
        return qk * encoding[0] + self.rotate_half(qk) * encoding[1]


class CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        b, _, _ = descriptors.shape
        qk, v = self.to_qk(descriptors), self.to_v(descriptors)

        indices = torch.arange(b, device=descriptors.device)
        swap = (indices // 2) * 2 + (1 - indices % 2)  # swap trick
        m = multi_head_attention_dispatch(qk, qk[swap], v[swap], self.num_heads)
        m = self.to_out(m)
        descriptors = descriptors + self.ffn(torch.concat([descriptors, m], 2))
        return descriptors


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(
        self, descriptors: torch.Tensor, encodings: torch.Tensor
    ) -> torch.Tensor:
        descriptors = self.self_attn(descriptors, encodings)
        return self.cross_attn(descriptors)


def sigmoid_log_double_softmax(
    similarities: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    certainties = F.logsigmoid(z[0::2]) + F.logsigmoid(z[1::2]).transpose(1, 2)
    scores0 = F.log_softmax(similarities, 2)
    scores1 = F.log_softmax(similarities, 1)
    scores = scores0 + scores1 + certainties
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.25
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.matchability = nn.Linear(dim, 1, bias=True)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """build assignment matrix from descriptors"""
        mdescriptors = self.final_proj(descriptors) / self.scale
        similarities = mdescriptors[0::2] @ mdescriptors[1::2].transpose(1, 2)
        z = self.matchability(descriptors)
        scores = sigmoid_log_double_softmax(similarities, z)
        return scores

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches_npu(scores: torch.Tensor, threshold: float):
    """obtain matches from a log assignment matrix [BxNxN]"""
    max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2)
    max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
    m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]

    indices = torch.arange(m0.shape[1], device=m0.device).expand_as(m0)
    mutual = indices == m1.gather(1, m0)
    mscores = max0.values[:, :, 0].exp()
    valid = mscores > threshold

    return m0, valid & mutual, mscores


def filter_matches_cpu(m0, mask, mscores):
    b_idx, m0_idx = torch.where(mask)
    m1_idx = m0[b_idx, m0_idx]
    matches = torch.concat([b_idx[:, None], m0_idx[:, None], m1_idx[:, None]], 1)
    mscores = mscores[b_idx, m0_idx]
    return matches, mscores


class LightGlue(nn.Module):
    def __init__(
        self,
        url: str,
        input_dim: int = 256,
        descriptor_dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 9,
        filter_threshold: float = 0.1,  # match threshold
        depth_confidence: float = -1,  # -1 is no early stopping, recommend: 0.95
        width_confidence: float = -1,  # -1 is no point pruning, recommend: 0.99
    ) -> None:
        super().__init__()

        self.descriptor_dim = descriptor_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.filter_threshold = filter_threshold
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence

        if input_dim != self.descriptor_dim:
            self.input_proj = nn.Linear(input_dim, self.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        self.posenc = LearnableFourierPositionalEncoding(
            2, self.descriptor_dim, self.num_heads
        )

        d, h, n = self.descriptor_dim, self.num_heads, self.n_layers

        self.transformers = nn.ModuleList([TransformerLayer(d, h) for _ in range(n)])

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n - 1)]
        )
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor([self.confidence_threshold(i) for i in range(n)]),
        )

        state_dict = torch.hub.load_state_dict_from_url(url)

        # rename old state dict entries
        for i in range(n):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        # united_kpts_and_desc: torch.Tensor,  # (2B, N, 2+D), normalized
        keypoints,
        descriptors,
    ):
        # keypoints = united_kpts_and_desc[:, :, :2]
        # descriptors = united_kpts_and_desc[:, :, 2:]
        descriptors = self.input_proj(descriptors)

        # positional embeddings
        encodings = self.posenc(keypoints)  # (2, 2B, *, 64, 1)

        # GNN + final_proj + assignment
        for i in range(self.n_layers):
            # self+cross attention
            descriptors = self.transformers[i](descriptors, encodings)

        scores = self.log_assignment[self.n_layers - 1](descriptors)  # (B, N, N)

        # max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2)
        # max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
        # m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]

        # indices = torch.arange(m0.shape[1], device=m0.device).expand_as(m0)
        # mutual = indices == m1.gather(1, m0)
        # mscores = max0.values[:, :, 0].exp()
        # valid = mscores > self.filter_threshold

        # return m0, valid & mutual, mscores
        return scores

        # m0, mask, mscores = filter_matches_npu(scores, self.filter_threshold)
        # return scores, keypoints

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)
