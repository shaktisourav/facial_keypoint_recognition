This folder is from assignment2 of CS231N class at Stanford.
We will be using this implementation to build our vanilla CNN.


Fast layers are implemented as a cython extension. You need to run the following:

python setup.py build_ext --inplace.


Installing on Windows? See
https://github.com/cython/cython/wiki/InstallingOnWindows
https://github.com/cython/cython/wiki/CythonExtensionsOnWindows