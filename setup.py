"""Build script for Cython extension."""

import os

from setuptools import setup, Extension

# Default: portable flags safe for pip wheels and cross-CPU distribution.
# Set ZLFI_NATIVE=1 for local builds to enable -march=native -ffast-math.
_portable_flags = ["-O3", "-fopenmp"]
_native_flags = ["-O3", "-march=native", "-ffast-math", "-fopenmp"]

compile_args = _native_flags if os.environ.get("ZLFI_NATIVE") == "1" else _portable_flags
link_args = ["-fopenmp"]

try:
    from Cython.Build import cythonize

    extensions = cythonize(
        [
            Extension(
                "zlfi._core",
                sources=["src/zlfi/_core.pyx"],
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
            "zlfi._core",
            sources=["src/zlfi/_core.c"],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
        )
    ]

setup(
    ext_modules=extensions,
)
