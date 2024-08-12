from setuptools import setup, find_packages

setup(
    name='torchtrainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    description='PyTorch Training Framework: A customizable PyTorch training loop class with support for metrics tracking, early stopping, and callbacks. Includes methods for multi-class and binary accuracy, precision, recall, and R² score calculations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='code_moon',
    author_email='paraglondhe123@gmail.com',
    url='https://github.com/paraglondhe098/torchtrainer.git',
)
