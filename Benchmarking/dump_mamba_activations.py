#!/usr/bin/env python3

import argparse
import os
from contextlib import contextmanager


def parse_int_list(text):
    values = []
    for item in (text or "").split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def torch_dtype_for_name(torch, dtype_name):
    if dtype_name == "fp32":
        return torch.float32
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype {dtype_name}")


def disable_fast_path(model):
    for layer in getattr(model.backbone, "layers", []):
        mixer = getattr(layer, "mixer", None)
        if hasattr(mixer, "use_fast_path"):
            mixer.use_fast_path = False


@contextmanager
def capture_selective_scan(target_call_index):
    from mamba_ssm.modules import mamba_simple as mamba_simple_mod
    from mamba_ssm.ops import selective_scan_interface as scan_interface

    original_simple = mamba_simple_mod.selective_scan_fn
    original_interface = scan_interface.selective_scan_fn
    captured = {}
    state = {"count": 0}

    def wrapped(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
        call_index = state["count"]
        state["count"] += 1
        if call_index == target_call_index:
            captured["u"] = u.detach().cpu()
            captured["delta"] = delta.detach().cpu()
            captured["A"] = A.detach().cpu()
            captured["B"] = B.detach().cpu()
            captured["C"] = C.detach().cpu()
            captured["D"] = None if D is None else D.detach().cpu()
            captured["z"] = None if z is None else z.detach().cpu()
            captured["delta_bias"] = None if delta_bias is None else delta_bias.detach().cpu()
            captured["delta_softplus"] = bool(delta_softplus)
        return original_simple(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)

    mamba_simple_mod.selective_scan_fn = wrapped
    scan_interface.selective_scan_fn = wrapped
    try:
        yield captured
    finally:
        mamba_simple_mod.selective_scan_fn = original_simple
        scan_interface.selective_scan_fn = original_interface


def derive_scalar_surrogate(torch, captured, length, dims, state_index):
    u = captured["u"][0].float()          # [D_inner, L]
    delta = captured["delta"][0].float()  # [D_inner, L]
    A = captured["A"].float()             # [D_inner, d_state]
    B = captured["B"][0].float()          # [d_state, L]
    delta_bias = captured["delta_bias"]
    if delta_bias is not None:
        delta = delta + delta_bias.float().unsqueeze(-1)
    if captured.get("delta_softplus"):
        delta = torch.nn.functional.softplus(delta)

    if state_index < 0 or state_index >= A.shape[1]:
        raise ValueError(f"state_index {state_index} out of range for d_state={A.shape[1]}")

    available_dim = u.shape[0]
    if any(dim > available_dim for dim in dims):
        raise ValueError(f"requested dims {dims} exceed available inner dimension {available_dim}")

    outputs = {}
    A_state = A[:, state_index].unsqueeze(-1)
    B_state = B[state_index].unsqueeze(0)
    a_full = torch.exp(delta * A_state)
    b_full = delta * B_state * u

    for dim in dims:
        a = a_full[:dim, :length].transpose(0, 1).contiguous()
        b = b_full[:dim, :length].transpose(0, 1).contiguous()
        outputs[dim] = (a, b)
    return outputs


def save_surrogate_inputs(torch, surrogate, out_dir, length):
    os.makedirs(out_dir, exist_ok=True)
    for dim, (a_tm, b_tm) in surrogate.items():
        path = os.path.join(out_dir, f"input_B1_L{length}_D{dim}.bin")
        with open(path, "wb") as f:
            f.write(a_tm.numpy().astype("float32", copy=False).tobytes())
            f.write(b_tm.numpy().astype("float32", copy=False).tobytes())


def main():
    parser = argparse.ArgumentParser(description="Dump real Mamba selective-scan activations and scalar surrogate inputs")
    parser.add_argument("--pretrained", default="state-spaces/mamba-2.8b")
    parser.add_argument("--lengths", default="2048,8192,32768")
    parser.add_argument("--dims", default="1,16,64,256,512")
    parser.add_argument("--layer_index", type=int, default=0)
    parser.add_argument("--state_index", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--full_out_dir", default="RealInputs/full_scan_tensors")
    parser.add_argument("--surrogate_out_dir", default="RealInputs/surrogate_inputs")
    args = parser.parse_args()

    import torch
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch_dtype_for_name(torch, args.dtype)
    lengths = parse_int_list(args.lengths)
    dims = parse_int_list(args.dims)

    model = MambaLMHeadModel.from_pretrained(args.pretrained, device=device, dtype=dtype)
    model.eval()
    disable_fast_path(model)

    os.makedirs(args.full_out_dir, exist_ok=True)
    os.makedirs(args.surrogate_out_dir, exist_ok=True)

    for length in lengths:
        input_ids = torch.randint(0, model.config.vocab_size, (1, length), device=device)
        with torch.inference_mode():
            with capture_selective_scan(args.layer_index) as captured:
                model(input_ids)
        if not captured:
            raise RuntimeError(f"No selective_scan capture recorded for layer call index {args.layer_index}")

        full_path = os.path.join(args.full_out_dir, f"scan_inputs_layer{args.layer_index}_L{length}_{args.dtype}.pt")
        torch.save(captured, full_path)

        surrogate = derive_scalar_surrogate(torch, captured, length, dims, args.state_index)
        save_surrogate_inputs(torch, surrogate, args.surrogate_out_dir, length)
        print(f"saved full selective-scan tensors to {full_path}")
        for dim in dims:
            print(f"saved surrogate scalar inputs: {args.surrogate_out_dir}/input_B1_L{length}_D{dim}.bin")


if __name__ == "__main__":
    raise SystemExit(main())
