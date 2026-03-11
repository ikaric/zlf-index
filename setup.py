"""Build script for Cython extension."""

import os
import subprocess
import sys

from setuptools import setup, Extension


def _detect_openmp():
    """Return (compile_flags, link_flags) for OpenMP, or empty lists if unavailable."""
    if sys.platform == "darwin":
        # Apple clang lacks OpenMP; check for Homebrew libomp.
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL
            ).decode().strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            return [], []
        return (
            ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
            [f"-L{prefix}/lib", "-lomp"],
        )
    return ["-fopenmp"], ["-fopenmp"]


_omp_compile, _omp_link = _detect_openmp()

_portable_flags = ["-O3"] + _omp_compile
_native_flags = ["-O3", "-march=native", "-ffast-math"] + _omp_compile

compile_args = _native_flags if os.environ.get("SUFFIX25_NATIVE") == "1" else _portable_flags
link_args = list(_omp_link)

try:
    from Cython.Build import cythonize

    extensions = cythonize(
        [
            Extension(
                "suffix25._core",
                sources=["src/suffix25/_core.pyx"],
                extra_compile_args=compile_args,
                extra_link_args=link_args,
            )
        ],
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": 3,
        },
    )
except ImportError:
    extensions = [
        Extension(
            "suffix25._core",
            sources=["src/suffix25/_core.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]

setup(
    ext_modules=extensions,
)
