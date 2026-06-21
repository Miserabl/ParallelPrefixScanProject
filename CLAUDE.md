# Local Notes

## Optional Python baseline

The official `selective_scan_cuda` baseline is consumed as a Python dependency,
not vendored into this repo. It is tightly coupled to the local Torch/CUDA ABI.

Pinned dependency:

```bash
pip install -r requirements.txt --no-build-isolation
```

## Colab install recipe

Colab already ships with CUDA-enabled PyTorch, so install `mamba-ssm` without
build isolation:

```bash
pip install --upgrade pip
pip install mamba-ssm==2.3.2 --no-build-isolation
```

If the import fails on a given machine, the Python timing harness skips the
`selective_scan_cuda` baseline gracefully and leaves the C++ benchmark flow
usable.
