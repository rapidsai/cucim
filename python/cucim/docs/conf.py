# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os

# Versioning
with open("../../../VERSION") as f:
    version_long = f.readline().strip()

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.ifconfig',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'myst_nb',
    'sphinx_copybutton',
    'sphinx_togglebutton',
    'sphinx_panels',
    'ablog',
    'sphinxemoji.sphinxemoji',
]
# source_suffix = {
#     '.rst': 'restructuredtext',
#     '.ipynb': 'myst-nb',
#     '.myst': 'myst-nb',
# }
master_doc = 'index'
project = 'cuCIM'
year = '2020-2021'
author = 'NVIDIA'
copyright = '{0}, {1}'.format(year, author)
version = release = version_long

pygments_style = 'trac'
templates_path = ['.']
extlinks = {
    'issue': ('https://github.com/rapidsai/cucim/issues/%s', '#'),
    'pr': ('https://github.com/rapidsai/cucim/pull/%s', 'PR #'),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if not on_rtd:  # only set the theme if we're building docs locally
    html_theme = 'pydata_sphinx_theme'  # 'sphinx_book_theme'
    # https://github.com/pandas-dev/pydata-sphinx-theme
    # https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/index.html

html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = False
# html_sidebars = {
#     '**': ['searchbox.html', 'globaltoc.html', 'sourcelink.html'],
# }
html_short_title = '%s-%s' % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

html_show_sourcelink = True

# Options for linkcheck builder
#
# Reference
# : https://www.sphinx-doc.org/en/master/usage/configuration.html?highlight=linkcheck#options-for-the-linkcheck-builder)  # noqa
linkcheck_ignore = [r'^\/', r'^\.\.']

# Options for sphinx.ext.todo
# (reference: https://www.sphinx-doc.org/en/master/usage/extensions/todo.html)

todo_include_todos = True

# Options for sphinxemoji.sphinxemoji
# (reference: https://sphinxemojicodes.readthedocs.io/en/stable/#supported-codes)  # noqa


# Options for myst
# (reference: https://myst-parser.readthedocs.io/en/latest/index.html)

# https://myst-parser.readthedocs.io/en/latest/using/syntax-optional.html#markdown-figures  # noqa
myst_enable_extensions = ["colon_fence"]

# Options for pydata-sphinx-theme
# (reference: https://pydata-sphinx-theme.readthedocs.io/en/latest/user_guide/configuring.html)  # noqa

html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

html_theme_options = {
    "external_links": [
        {
            "name": "Submit Issue",
            "url": "https://github.com/rapidsai/cucim/issues/new/choose",  # noqa
        }
    ]
}

# Options for Sphinx Book Theme
# (reference: https://github.com/executablebooks/sphinx-book-theme/blob/master/setup.py)  # noqa

# html_theme_options = {
#     "repository_url": "https://github.com/rapidsai/cucim",
#     "use_repository_button": True,
#     "use_issues_button": True,
#     #"use_edit_page_button": True,
#     "repository_branch": "dev",
#     #"path_to_docs": "python/cucim/docs",
#     "home_page_in_toc": True,
# }

# Options for myst-nb
# (reference: https://myst-nb.readthedocs.io/en/latest/)

# Prevent the following error
#     MyST NB Configuration Error:
#    `nb_render_priority` not set for builder: doctest
nb_render_priority = {
    "doctest": ()
}

# Prevent creating jupyter_execute folder in dist
#  https://myst-nb.readthedocs.io/en/latest/use/execute.html#executing-in-temporary-folders  # noqa
execution_in_temp = True
jupyter_execute_notebooks = "off"
