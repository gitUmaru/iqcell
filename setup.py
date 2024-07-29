import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iqcell",
    version="0.1.0",
    author="Muhammad Umar Ali",
    author_email="umaruali@student.ubc.ca",
    description="A platform for predicting the effect of gene perturbations on developmental trajectories using single-cell RNAseq data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gitUmaru/iqcell",
    packages=setuptools.find_packages(),
    package_data={'exaamples': ['data/*.csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scikit-learn",
        "torch",
        "anndata",
        "tqdm",
        "scanpy"
   ]
)
