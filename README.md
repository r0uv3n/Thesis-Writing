# Repository for Bachelor's Thesis - Harmonic Functions and the Mass of Asymptotically Flat Half Spaces

This repository contains the $\LaTeX$ and `Python` source code for my Bachelor's Thesis.

## How to run the code for computations and graphs

Within the Folder `Computations and Graphs`, there is both a Jupyter Notebook `computations_and_graphs.ipynb` and a corresponding Python script `computations_and_graphs.py`.

- To run the Jupyter Notebook, just execute it in some Jupyter Environment running a Python 3.10 kernel (any requirements should automatically be installed). This can e.g. be done on [the university's cloud Jupyter offering](computations_and_graphs.ipynb) by uploading and running the `.ipynb` file (just be sure to select the kernel "`Python 3.10 (XPython)`" and to recreate the folder structure of this project (in particular there should be a `figures` folder in the same directory as the `computations_and_graphs` folder)).
- To run the Python script:
  1. `git clone` this repository from the command line and `cd` into it.
  2. Create a virtual environment (with Python version 3.10). Assuming Python 3.10 is installed, running the following should work in Windows Powershell:

  ```powershell
  py -3.10 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install -r .\computations_and_graphs\requirements.txt
  ```

  On Linux in e.g. `bash`, one can probably use the following:

  ```bash
  python3.10 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r computations_and_graphs/requirements.txt
  ```

  3. Run

  ```bash
  python computations_and_graphs/computations_and_graphs.py
  ```

  4. Output should appear in the terminal and in the `figures` folder.

## How to compile the $\LaTeX$ source code

This thesis was written using `MikTeX` and my custom package [`hrftex`](https://github.com/r0uv3n/hrftex). After following the installation instructions [here](https://github.com/r0uv3n/hrftex#installation-via-github-for-miktex), it should be possible to compile the file

```filename
Harmonic_Functions_and_the_Mass_of_Asymptotically_Flat_Half_Spaces.tex
```

using e.g. `pdflatex` or `latexmk`.
