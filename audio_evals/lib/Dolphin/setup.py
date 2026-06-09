import os
import pkg_resources
from setuptools import find_packages, setup


def read_version(fname="dolphin/version.py"):
    exec(compile(open(fname, encoding="utf-8").read(), fname, "exec"))
    return locals()["__version__"]


requirements = []
setup(
    name="dataoceanai-dolphin",
    py_modules=["dolphin"],
    version=read_version(),
    description="DataoceanAI Open-source Large Speech Model",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.7",
    author="DataoceanAI",
    url="https://github.com/DataoceanAI/dolphin",
    license="Apache-2.0",
    license_files="LICENSE",
    packages=[
        "dolphin",
        "dolphin/assets",
    ],
    package_data={"dolphin/assets": ["*"]},
    install_requires=requirements
    + [
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    entry_points={
        "console_scripts": ["dolphin=dolphin.transcribe:cli"],
    },
    include_package_data=True,
    extras_require={"dev": ["pytest", "flake8"]},
)
