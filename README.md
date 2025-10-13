# Coastal Hazards Toolkit - Tiling

The **Coastal Hazards Toolkit - Tiling (cht-tiling)** is a Python package that provides utilities for creating and managing tiled datasets used in coastal hazard analysis and visualization workflows.

## Features
- Efficient tiling of large coastal hazard datasets.
- Tools for generating multi-resolution tile structures.
- Built-in support for integration with the Coastal Hazards Toolkit ecosystem.

## Installation
You can install the package from PyPI:

```bash
pip install cht-tiling
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv add cht-tiling
```

## Development
Clone the repository and install with development dependencies:

```bash
uv sync --all-extras --dev
```

Run the test suite:

```bash
uv run pytest
```

Build and publish to PyPi:
```bash
uv build
uv publish
```

## License
This project is licensed under the [MIT License](LICENSE).
