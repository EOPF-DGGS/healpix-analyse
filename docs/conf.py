# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "healpix-analyse"
copyright = "2024, Jean-Marc Delouis, Tina Odaka"
author = "Jean-Marc Delouis, Tina Odaka"
release = "0.1.0"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "autoapi.extension",       # API docs from source code
    "sphinx.ext.napoleon",     # NumPy / Google docstrings
    "sphinx.ext.viewcode",     # Links to source code
    "sphinx.ext.intersphinx",  # Cross-references to external docs
    "sphinx.ext.mathjax",      # Math rendering
    "myst_nb",                 # Markdown + Jupyter notebooks (inclut myst_parser)
]

# MyST / myst-nb configuration
myst_enable_extensions = [
    "amsmath",       # LaTeX math blocks
    "colon_fence",   # ::: directive fences
    "dollarmath",    # $...$ inline math
    "deflist",       # definition lists
]
nb_execution_mode = "off"  # don't execute notebooks at build time

# ---------------------------------------------------------------------------
# AutoAPI
# ---------------------------------------------------------------------------
autoapi_dirs = ["../healpix_analyse"]
autoapi_type = "python"
autoapi_output_dir = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # NOTE: "imported-members" is intentionally omitted — it generates
    # hundreds of duplicate cross-reference warnings.
]
autoapi_keep_files = True
autoapi_python_use_implicit_namespaces = True  # parse AST only, don't import

# ---------------------------------------------------------------------------
# Suppress known harmless warnings
# ---------------------------------------------------------------------------
suppress_warnings = [
    "autoapi.python_import_resolution",
    "autoapi",
    "myst.header",
    "ref.python",
    "intersphinx.external",
]

# ---------------------------------------------------------------------------
# Napoleon (NumPy docstrings)
# ---------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output — PyData theme (same as healpix-geo, numpy, scipy…)
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "navigation_depth": 4,
    "show_toc_level": 2,
    "github_url": "https://github.com/EOPF-DGGS/healpix-analyse",
    "navbar_end": ["navbar-icon-links"],
    "footer_start": ["copyright"],
}

html_title = "healpix-analyse"
html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# Source suffixes — laisser myst_parser et myst_nb les enregistrer eux-mêmes
# ---------------------------------------------------------------------------
# Ne pas déclarer manuellement "myst" ou "myst-nb" comme parsers :
# myst_parser enregistre automatiquement .md, myst_nb enregistre .ipynb.
# Une déclaration manuelle ici provoquerait "Source parser for myst not registered".
source_suffix = [".rst", ".md", ".ipynb"]
