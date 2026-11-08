# panther-em

[![License](https://img.shields.io/pypi/l/panther-em.svg?color=green)](https://github.com/mgiammar/panther-em/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/panther-em.svg?color=green)](https://pypi.org/project/panther-em)
[![Python Version](https://img.shields.io/pypi/pyversions/panther-em.svg?color=green)](https://python.org)
[![CI](https://github.com/mgiammar/panther-em/actions/workflows/ci.yml/badge.svg)](https://github.com/mgiammar/panther-em/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mgiammar/panther-em/branch/main/graph/badge.svg)](https://codecov.io/gh/mgiammar/panther-em)

**P**ipelined **A**cceleratio**N** of **T**emplate matc**H**ing via **E**igendecomposition of **R**otational projections in cryo-**EM**

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork mgiammar/panther-em --clone
# or just
# gh repo clone mgiammar/panther-em
cd panther-em
uv sync
```

### Installation

The ``uv`` tool should automatically handle installation and dependencies.

```sh
uv venv create
source .venv/bin/activate
uv pip install -e .
```

Sometimes, ``uv`` may not play nice with the ``cuCIM`` dependency for GPU acceleration. In that case, you may try installing through a conda environment:

```sh
# Update desired conda env name and python version
conda create -n panther-em python=3.13
conda activate panther-em

# 'cuda13' dependency, can also use 'cuda12'
uv pip install -e '.[cuda13]`
```

### Optional dependencies

Panther-EM includes optional dependencies for GPU acceleration and custom CUDA kernels which may be installed via
```sh
uv pip install -e '.[cuda13,fused-kernels]'
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
