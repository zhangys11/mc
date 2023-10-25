from setuptools import setup

setup(
    name="mc-tk",
    version="1.1",
    packages=["mc"],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
        "IPython",
        "tqdm",
        "scikit-learn"
    ],
)
