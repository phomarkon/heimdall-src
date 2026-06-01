.PHONY: figures-se figures-clean lint-arch help

help:
	@echo "make figures-se    regenerate Chapter 2 software-engineering figures"
	@echo "make lint-arch     enforce architectural layering via import-linter"

figures-se:
	uv run python research/scripts/build_figures.py

lint-arch:
	uv run lint-imports
