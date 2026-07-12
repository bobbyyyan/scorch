# Minimal makefile for Sphinx documentation.
#
# Common targets:
#   make html      build the HTML site into build/html
#   make livehtml  auto-rebuilding preview server (needs sphinx-autobuild)
#   make linkcheck verify external links
#   make clean     remove the build directory

SPHINXOPTS    ?= -W --keep-going -n
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html dirhtml singlehtml linkcheck clean livehtml Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Live-reloading preview: `pip install sphinx-autobuild` then `make livehtml`.
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O) \
		--ignore "*/_kb/*" --open-browser

clean:
	rm -rf "$(BUILDDIR)"

# Catch-all: route everything else to sphinx-build (html, linkcheck, ...).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
