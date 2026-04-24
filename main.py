"""Repo-root wrapper for the packaged CLI entrypoint."""

from smooth_life_search.input.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
