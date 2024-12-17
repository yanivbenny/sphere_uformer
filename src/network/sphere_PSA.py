from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from network.position_encoding import RelativePositionBias
from trimesh_utils import IcoSphereRef


class SphereSelfAttention(nn.Module):
    LOGIT_SCALE_PRE_REL_BIAS: bool = True

    def __init__(
            self, *,
            rank: int,
            icosphere_ref: IcoSphereRef,
            win_size_coef,
            num_heads,
            d_model,
            d_head_coef,
            qkv_bias,
            # use_v_proj,
            attn_drop=0.,
            out_drop=0.,
            abs_pos_enc: bool = False,
            abs_pos_enc_size: int = 0,
            rel_pos_bias: bool = False,
            rel_pos_bias_size: int = 0,
            rel_pos_init_variance: float = 0.0,
            append_self: bool = False,
    ):
        """
        :param num_heads: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        :param num_keys: number of keys
        """

        super().__init__()

        self.rank = rank
        self.icosphere_ref = icosphere_ref
        self.win_size_coef = win_size_coef

        self.apply_rel_pos_bias = rel_pos_bias
        self.rel_pos_bias = RelativePositionBias(rank, icosphere_ref, win_size_coef, rel_pos_bias_size=rel_pos_bias_size, num_heads=num_heads, init_variance=rel_pos_init_variance)
        self.num_keys = self.rel_pos_bias.num_keys

        # We assume d_v always equals d_k, d_q = d_k = d_v = d_m // h
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = (d_model // num_heads) * d_head_coef
        assert self.d_head == int(self.d_head)

        self.append_self = append_self

        self.apply_abs_pos_enc = abs_pos_enc
        if self.apply_abs_pos_enc:
            self.q_abs_pos_proj = nn.Linear(abs_pos_enc_size, self.d_head * num_heads, bias=False)
            self.k_abs_pos_proj = nn.Linear(abs_pos_enc_size, self.d_head * num_heads, bias=False)

        self.q_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, self.d_head * num_heads, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, self.d_head * (num_heads + append_self), bias=qkv_bias)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)
        # self.logit_scale_pre_rel_bias = logit_scale_pre_rel_bias

        self.out_proj = nn.Linear(self.d_head * (num_heads + append_self), d_model)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_drop = nn.Dropout(out_drop)

    def forward(
            self,
            x: Tensor,
            pos: Optional[Tensor],
            query_mask: Tensor = None,
            key_masks: Optional[Tensor] = None,

    ):
        """
        :param x: B, D, C
        :param query_mask:
        :param key_masks:
        :return:
        """
        # if key_masks is None:
        #     key_masks = [None] * len(keys)

        metadata = None

        N, D, C = x.shape
        H = self.num_heads
        K = self.num_keys
        C_H = self.d_head

        # q,k,v: [N, H, D, C//H]
        q = self.q_proj(x).view(N, D, H, C_H).permute(0,2,1,3)
        k = self.k_proj(x).view(N, D, H, C_H).permute(0,2,1,3)
        v = self.v_proj(x).view(N, D, H + self.append_self, C_H).permute(0,2,1,3)
        if self.append_self:
            v, v_self = torch.split(v, (H, 1), dim=1)
            v_self = v_self.squeeze(1)

        if self.apply_abs_pos_enc:
            assert D == pos.shape[1]
            q_pos = self.q_abs_pos_proj(pos).view(1, D, H, C_H).permute(0,2,1,3)
            k_pos = self.k_abs_pos_proj(pos).view(1, D, H, C_H).permute(0,2,1,3)
            q = q + q_pos
            k = k + k_pos

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # aligned: [N, H, D, K, C//H]
        # expanded_idx = self.idx[None, None, :, :, None].expand(N, H, -1, -1, C_H)
        # expanded_idx_mask = self.idx_mask[None, None, :, :].expand(N, H, -1, -1)
        expanded_idx = self.rel_pos_bias.idx.expand(N, H, -1, -1, C_H)
        expanded_idx_mask = self.rel_pos_bias.idx_mask.expand(N, H, -1, -1)
        expanded_k = k[:, :, :, None, :].expand(-1, -1, -1, K, -1)
        expanded_v = v[:, :, :, None, :].expand(-1, -1, -1, K, -1)

        aligned_k = torch.gather(expanded_k, dim=2, index=expanded_idx)
        aligned_v = torch.gather(expanded_v, dim=2, index=expanded_idx)

        # (q: [N, H, D, C_H] , aligned_k: [N, H, D, K, C_H]) -> attn: [N, H, D, K]
        attn = (q[:, :, :, None, :] * aligned_k).sum(-1)

        if self.LOGIT_SCALE_PRE_REL_BIAS:
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

        if self.apply_rel_pos_bias:
            rel_coords, rel_bias = self.rel_pos_bias(aligned_k)
            attn = attn + rel_bias

        if not self.LOGIT_SCALE_PRE_REL_BIAS:
            logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
            attn = attn * logit_scale

        # mask for cases where number of keys < max number of keys
        attn = torch.masked_fill(attn, mask=~expanded_idx_mask, value=float('-inf'))

        # B, H, D, K
        attn = F.softmax(attn, dim=-1)

        # mask nan position
        if query_mask is not None:
            raise NotImplementedError("No support for query_mask")
            # B, H, W, 1, 1
            query_mask_ = query_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            attn = torch.masked_fill(attn, query_mask_.expand_as(attn), 0.0)

        attn = self.attn_drop(attn)

        # (attn: [N, H, D, K] , aligned_v: [N, H, D, K, C_H]) -> out: [N, H, D, C_H]
        out = (attn.unsqueeze(-1) * aligned_v).sum(-2)
        # out = out.permute(0, 2, 1, 3).reshape(N, D, C)
        out = einops.rearrange(out, "N H D C_H -> N D (H C_H)")
        if self.append_self:
            out = torch.cat((out, v_self), dim=-1)
        out = self.out_proj(out)
        out = self.out_drop(out)

        return out
