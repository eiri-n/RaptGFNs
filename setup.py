from setuptools import setup, find_packages
import os

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

desc = "DNAアプタマー生成のためのMulti-Objective GFlowNets" 

setup(
    name="raptgfn",
    version="0.1.0",
    description=desc,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Modified for DNA Aptamer Generation",
    url="https://github.com/GFNOrg/multi-objective-gfn",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch>=2.0.0',
        'numpy',
        'matplotlib',
        'hydra-core>=1.3.2',
        'omegaconf',
        'tqdm',
        'cachetools',
        'wandb',
        'pymoo>=0.5.0',
        'cvxopt>=1.3.0',
        'plotly',
        'scipy',
    ],
    python_requires='>=3.8',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'raptgfn=raptgfn.main:main',
        ],
    },
)
