from setuptools import setup, find_packages
import re

try:
    import torch
    has_dev_pytorch = "dev" in torch.__version__
except ImportError:
    has_dev_pytorch = False

# Base equirements
install_requires = [
    "torch",
]

if has_dev_pytorch:  # Remove the PyTorch requirement
    install_requires = [
        install_require for install_require in install_requires
        if "torch" != re.split(r"(=|<|>)", install_require)[0]
    ]

setup(
    name='RAdam',
    version='0.0.1',
    url='https://github.com/LiyuanLucasLiu/RAdam.git',
    author='Liyuan Liu',
    author_email='llychinalz@gmail.com',
    description='Implementation of the RAdam optimization algorithm described in On the Variance of the Adaptive Learning Rate and Beyond (https://arxiv.org/abs/1908.03265)',
    packages=find_packages(),
    install_requires=install_requires,
)
