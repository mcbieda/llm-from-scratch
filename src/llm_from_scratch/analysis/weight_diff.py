import re
import math
import torch
import pandas as pd


def tensor_stats(base_tensor, dapt_tensor, eps=1e-12):
    b = base_tensor.detach().float().cpu()
    d = dapt_tensor.detach().float().cpu()
    delta = d - b

    base_norm = torch.norm(b).item()
    dapt_norm = torch.norm(d).item()
    delta_norm = torch.norm(delta).item()

    rel_l2 = delta_norm / (base_norm + eps)
    mean_abs_delta = delta.abs().mean().item()
    max_abs_delta = delta.abs().max().item()

    b_flat = b.reshape(-1)
    d_flat = d.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        b_flat.unsqueeze(0), d_flat.unsqueeze(0), dim=1
    ).item()

    return {
        "base_norm": base_norm,
        "dapt_norm": dapt_norm,
        "delta_norm": delta_norm,
        "rel_l2": rel_l2,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "cosine_similarity": cos_sim,
        "numel": b.numel(),
    }


def get_state_dict(model):
    if hasattr(model, "module"):
        model = model.module
    return model.state_dict()


def compare_models(base_model, dapt_model, ignore_tied_lm_head=True):
    base_sd = get_state_dict(base_model)
    dapt_sd = get_state_dict(dapt_model)

    common_keys = sorted(set(base_sd.keys()) & set(dapt_sd.keys()))
    rows = []

    for k in common_keys:
        if ignore_tied_lm_head and k == "lm_head.weight":
            continue

        b = base_sd[k]
        d = dapt_sd[k]

        if b.shape != d.shape:
            continue

        if not torch.is_floating_point(b):
            continue

        stats = tensor_stats(b, d)
        stats["name"] = k
        rows.append(stats)

    df = pd.DataFrame(rows)
    return df[[
        "name", "numel", "rel_l2", "delta_norm",
        "mean_abs_delta", "max_abs_delta",
        "cosine_similarity", "base_norm", "dapt_norm"
    ]].sort_values("rel_l2", ascending=False).reset_index(drop=True)


def summarize_by_gpt2_block(param_df):
    def block_name(param_name):
        m = re.match(r"transformer\.h\.(\d+)\.", param_name)
        if m:
            return f"transformer.h.{m.group(1)}"
        elif param_name.startswith("transformer.wte"):
            return "transformer.wte"
        elif param_name.startswith("transformer.wpe"):
            return "transformer.wpe"
        elif param_name.startswith("transformer.ln_f"):
            return "transformer.ln_f"
        elif param_name.startswith("lm_head"):
            return "lm_head"
        else:
            return "other"

    df = param_df.copy()
    df["block"] = df["name"].apply(block_name)

    summary = (
        df.groupby("block", as_index=False)
          .agg(
              total_numel=("numel", "sum"),
              mean_rel_l2=("rel_l2", "mean"),
              max_rel_l2=("rel_l2", "max"),
              mean_abs_delta=("mean_abs_delta", "mean"),
              mean_cosine_similarity=("cosine_similarity", "mean"),
          )
          .sort_values("mean_rel_l2", ascending=False)
          .reset_index(drop=True)
    )
    return summary


def summarize_by_submodule(param_df):
    def submodule_name(param_name):
        patterns = [
            "attn.c_attn",
            "attn.c_proj",
            "mlp.c_fc",
            "mlp.c_proj",
            "ln_1",
            "ln_2",
            "wte",
            "wpe",
            "ln_f",
        ]
        for p in patterns:
            if p in param_name:
                return p
        return "other"

    df = param_df.copy()
    df["submodule"] = df["name"].apply(submodule_name)

    summary = (
        df.groupby("submodule", as_index=False)
          .agg(
              total_numel=("numel", "sum"),
              mean_rel_l2=("rel_l2", "mean"),
              max_rel_l2=("rel_l2", "max"),
              mean_abs_delta=("mean_abs_delta", "mean"),
              mean_cosine_similarity=("cosine_similarity", "mean"),
          )
          .sort_values("mean_rel_l2", ascending=False)
          .reset_index(drop=True)
    )
    return summary

def summarize_by_block_proper(param_df, base_model, dapt_model):
    base_sd = get_state_dict(base_model)
    dapt_sd = get_state_dict(dapt_model)

    block_to_keys = {}

    for k in param_df["name"]:
        m = re.match(r"transformer\.h\.(\d+)\.", k)
        if m:
            block = f"transformer.h.{m.group(1)}"
        elif k.startswith("transformer.wte"):
            block = "transformer.wte"
        elif k.startswith("transformer.wpe"):
            block = "transformer.wpe"
        elif k.startswith("transformer.ln_f"):
            block = "transformer.ln_f"
        else:
            continue

        block_to_keys.setdefault(block, []).append(k)

    rows = []
    for block, keys in block_to_keys.items():
        base_sq = 0.0
        delta_sq = 0.0
        total_numel = 0

        for k in keys:
            b = base_sd[k].detach().float().cpu()
            d = dapt_sd[k].detach().float().cpu()
            delta = d - b

            base_sq += torch.sum(b * b).item()
            delta_sq += torch.sum(delta * delta).item()
            total_numel += b.numel()

        rel_l2_block = math.sqrt(delta_sq) / (math.sqrt(base_sq) + 1e-12)
        rows.append({
            "block": block,
            "total_numel": total_numel,
            "block_rel_l2": rel_l2_block,
        })

    return pd.DataFrame(rows).sort_values("block_rel_l2", ascending=False).reset_index(drop=True)