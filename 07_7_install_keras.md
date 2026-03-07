# Install Keras

Keras (Python package) is installed by installing Tensorflow. Keras recently updated from version 2 to version 3. We're going to install Keras version 3 for R because it's by far the smoothest installation and only minor updates are needed for the R code examples. For Python users (i.e. not via R), to keep everything stable and working with the textbook,  we'll install Keras version 2 (via Tensorflow version 2.13).



## R users

The entire installation can be done from within R. Start R and run:

```R
install.packages("keras3")
```

For the latest version of the R `keras3` package, according to the documentation this should "just work". In other words, you should be able to run any R keras code now. Let me know if this does indeed work!

However, your mileage may differ. It didn't work for me on Linux. If the above doesn't just work, from within R also run the following to complete a full installation:

```R
reticulate::install_python(version = "3.11:latest")
keras::install_keras(python_version = "3.11")
```

This will install Python version 3.11.* into a Python virtual environment and install compatible Keras and Tensorflow versions into a Python virtual environment within your R installation. If Keras doesn't work at this stage, please let me know. There may be some warnings with these installations. Record the warning but if Keras in R works you can probably safely ignore it.



## Python users

First [install miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/) if you haven't already (you can also use an equivalent such as Anaconda or Mambaforge).

Then open a terminal (on Windows 10/11 you want the miniconda terminal app) and run:

```bash
conda create --name py-tensorflow
conda activate py-tensorflow
conda install python=3.12 -c conda-forge
pip install tensorflow pandas matplotlib plotnine
```

The official tensorflow/keras install instructions advise using `pip install` within the conda environment, rather than use `conda install`, because the conda repositories are often not up to date. You need to `pip install` all packages at once, otherwise later package installs could update keras and tensorflow as dependencies, which will break things. 

To use Keras and run the class Python code, within your IDE you need to start Python out of this environment before running code (e.g. in Positron, click `Start Session` and choose the `py-tensorflow` instance of Python).
