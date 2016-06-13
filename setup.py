from distutils.core import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules = cythonize(["seq2seq_model.pyx", "data_utils.pyx", 'translate.pyx']), include_dirs=[numpy.get_include()]
)