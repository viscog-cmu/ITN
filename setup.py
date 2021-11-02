import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="topographic",
    version="1.0.0",
    author="Nicholas M. Blauch",
    author_email="blauch@cmu.edu",
    description="Simulations of topographically organized DNN models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/viscog-cmu/ITN",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)