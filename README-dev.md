# Readme for workshop authors / developers

We develop the presentation and notebooks with [rise](https://rise.readthedocs.io/en/stable/). 
This way the code is directly next to the explanations which makes it easier for the audience to follow.

Notebooks can be converted to an html slide show from which a pdf can be extracted. Unfortunately, extracting
the pdf currently cannot be automated. The instructions for extracting the pdf are given 
[here](https://rise.readthedocs.io/en/stable/exportpdf.html).

## Setup

We recommend to install rise with conda (installation with pip may cause problems). We also use the
[spellchecker](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/spellchecker/README.html)
and [equation-numbering](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/equation-numbering/readme.html)
extensions. 

To configure everything, activate a conda env and run
```bash
conda install -c conda-forge notebook rise jupyter_contrib_nbextensions jupyter_nbextensions_configurator
python ./configure_spellcheck_dict.py
jupyter nbextension enable spellchecker/main
jupyter nbextension enable equation-numbering/main
```

Use the extension-configurator for customizing your slideshow as described 
[here](https://rise.readthedocs.io/en/stable/customize.html).

## Limitations

1. Rise does not work with jupyterlab, the creator is (slowly) working on this issue
2. Manual pdf export through chrome is really a bit annoying
3. Equation referencing does not work together with rise, inside presentation/pdf hyperlinks are broken.
   We might want to contribute to rise to make it work if we settle on this technology stack.
4. Prettyness of slideshow depends on screen size as far as I can tell.
