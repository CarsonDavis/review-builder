# Coding Notes
This repo has various built in tests and checks to ensure the code is clean and working as expected. Some things, such as pre-commit hooks, will need to be installed before contributing code.

## Type checks

Running type checks with mypy:
```bash
$ mypy book_summarizer
```

## Running tests with pytest
```bash
$ pytest
```

## Pre-Commit hook instructions

Hooks have to be run on every commit to automatically take care of linting and structuring.

To install pre-commit package manager :
```bash
$ pip install pre-commit
```

Install the git hook scripts:
```bash
$ pre-commit install
```

Run against the files:
```bash
$ pre-commit run --all-files
```
It's usually a good idea to run the hooks against all of the files when adding new hooks (usually `pre-commit` will only run on the changes files during git hooks).
