import re
import math
import torch
import pandas as pd


def tensor_stats(base_tensor, dapt_tensor, eps=1e-12):
    base_tensor = base_tensor.detach().float().cpu()
    dapt_tensor = dapt_tensor.detach().float().cpu()
    delta_tensor = dapt_tensor - base_tensor

    base_norm = torch.norm(base_tensor).item()
    dapt_norm = torch.norm(dapt_tensor).item()
    delta_norm = torch.norm(delta_tensor).item()

    rel_l2 = delta_norm / (base_norm + eps)
    mean_abs_delta = delta_tensor.abs().mean().item()
    max_abs_delta = delta_tensor.abs().max().item()

    base_flat = base_tensor.reshape(-1)
    dapt_flat = dapt_tensor.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        base_flat.unsqueeze(0), dapt_flat.unsqueeze(0), dim=1
    ).item()

    return {
        "base_norm": base_norm,
        "dapt_norm": dapt_norm,
        "delta_norm": delta_norm,
        "rel_l2": rel_l2,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "cosine_similarity": cos_sim,
        "numel": base_tensor.numel(),
    }


def get_state_dict(model):
    if hasattr(model, "module"):
        model = model.module
    return model.state_dict()


def compare_models(
    base_model,
    dapt_model,
    ignore_tied_out_head=True,
    max_changed_tensors=None,
):
    base_sd = get_state_dict(base_model)
    dapt_sd = get_state_dict(dapt_model)

    common_keys = sorted(set(base_sd.keys()) & set(dapt_sd.keys()))
    rows = []

    for param_name in common_keys:
        if ignore_tied_out_head and param_name == "out_head.weight":
            continue

        base_tensor = base_sd[param_name]
        dapt_tensor = dapt_sd[param_name]

        if base_tensor.shape != dapt_tensor.shape:
            continue

        if not torch.is_floating_point(base_tensor):
            continue

        stats = tensor_stats(base_tensor, dapt_tensor)
        stats["name"] = param_name
        rows.append(stats)

    df = pd.DataFrame(rows)
    df = df[[
        "name", "numel", "rel_l2", "delta_norm",
        "mean_abs_delta", "max_abs_delta",
        "cosine_similarity", "base_norm", "dapt_norm"
    ]].sort_values("rel_l2", ascending=False).reset_index(drop=True)

    if max_changed_tensors is not None:
        if max_changed_tensors < 0:
            raise ValueError(
                f"max_changed_tensors must be non-negative or None, got {max_changed_tensors}"
            )
        df = df.head(max_changed_tensors).reset_index(drop=True)

    return df


def _block_name_from_param_name(param_name):
    match = re.match(r"trf_blocks\.(\d+)\.", param_name)
    if match:
        return f"trf_blocks.{match.group(1)}"
    if param_name.startswith("tok_emb"):
        return "tok_emb"
    if param_name.startswith("pos_emb"):
        return "pos_emb"
    if param_name.startswith("final_norm"):
        return "final_norm"
    if param_name.startswith("out_head"):
        return "out_head"
    if param_name.startswith("drop_emb"):
        return "drop_emb"
    return "other"


def summarize_by_gpt2_block(param_df):
    df = param_df.copy()
    df["block"] = df["name"].apply(_block_name_from_param_name)

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


def _submodule_name_from_param_name(param_name):
    patterns = [
        "att.W_query",
        "att.W_key",
        "att.W_value",
        "att.out_proj",
        "ff.layers.0",
        "ff.layers.2",
        "norm1",
        "norm2",
        "drop_shortcut",
        "tok_emb",
        "pos_emb",
        "final_norm",
        "out_head",
        "drop_emb",
    ]
    for pattern in patterns:
        if pattern in param_name:
            return pattern
    return "other"


def summarize_by_submodule(param_df):
    df = param_df.copy()
    df["submodule"] = df["name"].apply(_submodule_name_from_param_name)

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

    for param_name in param_df["name"]:
        block_name = _block_name_from_param_name(param_name)
        if block_name == "other":
            continue

        block_to_keys.setdefault(block_name, []).append(param_name)

    rows = []
    for block_name, param_names in block_to_keys.items():
        base_sq = 0.0
        delta_sq = 0.0
        total_numel = 0

        for param_name in param_names:
            base_tensor = base_sd[param_name].detach().float().cpu()
            dapt_tensor = dapt_sd[param_name].detach().float().cpu()
            delta_tensor = dapt_tensor - base_tensor

            base_sq += torch.sum(base_tensor * base_tensor).item()
            delta_sq += torch.sum(delta_tensor * delta_tensor).item()
            total_numel += base_tensor.numel()

        rel_l2_block = math.sqrt(delta_sq) / (math.sqrt(base_sq) + 1e-12)
        rows.append({
            "block": block_name,
            "total_numel": total_numel,
            "block_rel_l2": rel_l2_block,
        })

    return pd.DataFrame(rows).sort_values("block_rel_l2", ascending=False).reset_index(drop=True)
