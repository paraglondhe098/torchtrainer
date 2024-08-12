from setuptools import setup, find_packages

setup(
    name='torchtrainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    description='A custom pytorch training loop, with custom and in built Callbacks, better logging.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='code_moon',
    author_email='paraglondhe123@gmail.com',
    url='https://github.com/paraglondhe098/torchtrainer.git',
)
