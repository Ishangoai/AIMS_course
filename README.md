# AIMS SA Course

### Instructions for development:
- create new feature branch from `development` following this convention: "feature-branch-{your name}"
- open the branch in CodeSpaces, selecting the weakest compute specs
- in the CodeSpace bash terminal, type `source .venv/bin/activate` to activate your personal Python environment

Other tips:
- type `uv pip list` to see installed Python packages
- type `pytest` to run unit tests
- type `ruff check` to run PEP8 checks
- type `pyright` to run type-hinting checks
- type `ml` run run the module (see the `[project.scripts]` section in `pyproject.toml`)

All these checks are also run in GitHub Actions (CI pipelines) when a new PR is raised to merge into `development`

