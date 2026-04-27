# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable for autodoc (optional, autoapi does not need it)
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
    # Automatic API docs from source code
    "autoapi.extension",
    # NumPy / Google docstring support
    "sphinx.ext.napoleon",
    # Links to source code
    "sphinx.ext.viewcode",
    # Cross-references to numpy, python, torch docs
    "sphinx.ext.intersphinx",
    # Math support
    "sphinx.ext.mathjax",
    # Markdown support (MyST)
    "myst_parser",
    # Jupyter notebooks as pages
    "nbsphinx",
]

# MyST configuration — enable extra syntax
myst_enable_extensions = [
    "amsmath",       # LaTeX math blocks
    "colon_fence",   # ::: directive fences
    "deflist",       # definition lists
    "dollarmath",    # $...$ inline math
    "html_image",    # raw <img> tags
    "tasklist",      # - [ ] checklists
]

# ---------------------------------------------------------------------------
# AutoAPI — generates API reference from source code
# ---------------------------------------------------------------------------
autoapi_dirs = ["../healpix_analyse"]
autoapi_type = "python"
autoapi_output_dir = "api"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_template_dir = None
autoapi_keep_files = True
suppress_warnings = ["autoapi.python_import_resolution"]

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
napoleon_use_admonition_for_references = True

# ---------------------------------------------------------------------------
# Intersphinx — cross-reference external docs
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

# ---------------------------------------------------------------------------
# HTML output — ReadTheDocs theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "logo_only": False,
}

html_title = "healpix-analyse"
html_short_title = "healpix-analyse"

# If you have a logo, put it in docs/_static/ and uncomment:
# html_logo = "_static/logo.png"

html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# nbsphinx — Jupyter notebooks
# ---------------------------------------------------------------------------
nbsphinx_execute = "never"   # don't re-run notebooks at build time

# ---------------------------------------------------------------------------
# Source suffixes
# ---------------------------------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst",
}
