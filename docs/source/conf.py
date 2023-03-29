# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sphinx_rtd_theme
import conifer

project = 'conifer'
copyright = '2023, Sioni Summers'
author = 'Sioni Summers'
version = str(conifer.__version__)
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx_mdinclude',
  'sphinx.ext.autodoc',
  'sphinx.ext.autosummary',
  'sphinx-favicon',
  'sphinx_rtd_theme',
]
autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

favicons = [
  {
    'rel': 'icon',
    'sizes': '32x32',
    'href': 'https://ssummers.web.cern.ch/conifer/conifer_favicon.png',
    'type': 'image/png'
  }
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']