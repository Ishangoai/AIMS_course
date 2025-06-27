# AIMS SA Course

### Instructions for development:
- create new feature branch from `development` following this convention: "feature-branch-{your name}"
- open the branch in CodeSpaces, selecting the weakest compute specs


Other tips:
- type `source .venv/bin/activate` to activate your personal uv Python environment (keep like this)
- type `deactivate` in the CodeSpace bash terminal to decativate your uv Python environment
- type `which python` to see which Python interpreter is active
- type `uv pip list` to see installed Python packages
- type `uv run pytest` to run unit tests
- type `uv ruff check` to run PEP8 checks
- type `uv run pyright` to run type-hinting checks
- type `uv python find` to see which uv environment is currently active
- type `ml` run run the module (see the `[project.scripts]` section in `pyproject.toml`)

All these checks are also run in GitHub Actions (CI pipelines) when a new PR is raised to merge into `development`
