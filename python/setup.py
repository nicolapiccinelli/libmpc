import os
import setuptools
from setuptools.command.build_ext import build_ext
from pathlib import Path
import subprocess
import sys

class CMakeExtension(setuptools.Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())

class BuildCMakeExt(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:

        # Make sure the build directory exists
        build_directory = os.path.abspath(self.build_temp)
        os.makedirs(build_directory, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_directory}",
            "-DPYTHON_EXECUTABLE:FILEPATH=" + sys.executable,
            "-DPYBIND11_PYTHON_VERSION=" + ".".join(str(v) for v in sys.version_info[:3]),
        ]

        # Print python version
        print("Python version: ", ".".join(str(v) for v in sys.version_info[:3]))

        build_args = ["--config", "Release"]

        print("Build directory: ", build_directory)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_directory)
        print("CMake done")

        subprocess.check_call(["cmake", "--build", ".", "--config", "Release"] + build_args, cwd=build_directory)
        print("Build done")

        # Move from build temp to final position
        for ext in self.extensions:
            self.move_output(ext)

    def move_output(self, ext):
            build_temp = Path(self.build_temp).resolve()
            dest_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            source_path = build_temp / self.get_ext_filename(ext.name)
            dest_directory = dest_path.parents[0]
            dest_directory.mkdir(parents=True, exist_ok=True)
            self.copy_file(source_path, dest_path)

# Read long description from the project README
here = Path(__file__).resolve().parent
try:
    long_description = (here.parent / "README.md").read_text(encoding="utf-8")
except Exception:
    long_description = "MPC++ bindings for Python."

setuptools.setup(
    name="pympcxx",
    version="1.0.0",
    description="MPC++ bindings for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicola Piccinelli",
    license="MIT",
    url="https://altairlab.gitlab.io/optcontrol/libmpc/",
    project_urls={
        "Documentation": "https://altairlab.gitlab.io/optcontrol/libmpc/",
        "Source": "https://github.com/nicolapiccinelli/libmpc",
        "Issues": "https://github.com/nicolapiccinelli/libmpc/issues",
        "PyPI": "https://pypi.org/project/pympcxx/",
    },
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries",
    ],
    keywords=[
        "mpc",
        "model predictive control",
        "optimization",
        "control systems",
        "nlopt",
        "osqp",
        "pybind11",
        "c++",
    ],
    python_requires=">=3.8",
    ext_modules=[CMakeExtension(name="pympcxx", sourcedir=".")],
    cmdclass={"build_ext": BuildCMakeExt},
    zip_safe=False
)
