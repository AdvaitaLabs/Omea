"""Padded SDPA backend for NEAT's newer torch varlen API on torch 2.8.

Implements identical independent, non-causal scaled dot-product attention.
Weights, attention connectivity, and model computations are unchanged. Floating
point rounding and performance can differ from the released Flash kernel.
"""
import sys
import types
import torch
from torch.nn.functional import scaled_dot_product_attention


def varlen_attn(q, k, v, *, cu_seq_q, cu_seq_k, max_q, max_k):
    b = len(cu_seq_q) - 1
    assert len(cu_seq_k) == b + 1
    nq = cu_seq_q[1:] - cu_seq_q[:-1]
    nk = cu_seq_k[1:] - cu_seq_k[:-1]
    iq = torch.repeat_interleave(torch.arange(b, device=q.device), nq.long())
    ik = torch.repeat_interleave(torch.arange(b, device=k.device), nk.long())
    jq = torch.arange(len(q), device=q.device) - cu_seq_q.long()[iq]
    jk = torch.arange(len(k), device=k.device) - cu_seq_k.long()[ik]
    qp = q.new_zeros((b, max_q, q.shape[1], q.shape[2]))
    kp = k.new_zeros((b, max_k, k.shape[1], k.shape[2]))
    vp = v.new_zeros((b, max_k, v.shape[1], v.shape[2]))
    qp[iq, jq] = q; kp[ik, jk] = k; vp[ik, jk] = v
    mask = torch.arange(max_k, device=k.device)[None, :] < nk[:, None]
    result = scaled_dot_product_attention(qp.transpose(1, 2), kp.transpose(1, 2),
        vp.transpose(1, 2), attn_mask=mask[:, None, None, :], dropout_p=0.0, is_causal=False)
    return result.transpose(1, 2)[iq, jq]


def install():
    try:
        from torch.nn.attention.varlen import varlen_attn as native
        return 'native torch varlen'
    except ImportError:
        module = types.ModuleType('torch.nn.attention.varlen')
        module.varlen_attn = varlen_attn
        sys.modules[module.__name__] = module
        return 'padded torch SDPA compatibility backend'


if __name__ == '__main__':
    torch.manual_seed(7)
    q, k, v = torch.randn(8, 2, 16), torch.randn(9, 2, 16), torch.randn(9, 2, 16)
    cq, ck = torch.tensor([0, 3, 8]), torch.tensor([0, 7, 9])
    actual = varlen_attn(q, k, v, cu_seq_q=cq, cu_seq_k=ck, max_q=5, max_k=7)
    expected = []
    for i in range(2):
        a, b, c = q[cq[i]:cq[i+1]], k[ck[i]:ck[i+1]], v[ck[i]:ck[i+1]]
        weights = ((a.permute(1,0,2) @ b.permute(1,2,0)) / 4).softmax(-1)
        expected.append((weights @ c.permute(1,0,2)).permute(1,0,2))
    error = (actual - torch.cat(expected)).abs().max().item()
    assert error < 1e-6, error
    print({'independent_cross_attention_max_abs_error': error})
