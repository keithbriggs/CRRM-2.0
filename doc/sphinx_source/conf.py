# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os, pathlib
import sys

sys.path.insert(0, os.path.abspath("./../../CRRM/"))
sys.path.insert(0, os.path.abspath("./../../"))
#print(f"path={sys.path}")

# don't fail the build if these are not importable...
autodoc_mock_imports = ["numpy", "scipy", "matplotlib"]

# -- Project information -----------------------------------------------------

project = "CRRM"
copyright = "2025, Keith Briggs"
author = "Keith Briggs"
source_suffix = ".rst"
master_doc = "index"

# The full version, including alpha/beta/rc tags
release = "2.0"
today_fmt = "%Y-%m-%d %H:%M"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # needed for "Parameters" to be recognized
    #'sphinx.ext.viewcode',
    #'sphinxarg.ext', # pip3 install sphinx-argparse (non-maintained version)
]

napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# https://www.sphinx-doc.org/en/master/usage/theming.html

html_theme = "pyramid"  # 'pyramid' # 'haiku' #,  'bizstyle' #  'sphinxdoc' # 'alabaster' 'agogo' # 'classic' # sphinxdoc

html_theme_options = {
    "sidebarwidth": 400,
    #    'globaltoc_collapse': False, # https://github.com/sphinx-doc/sphinx/issues/7171
    #    'globaltoc_maxdepth': 3,
}
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
