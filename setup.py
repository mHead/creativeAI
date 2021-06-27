""" From the folder immediately above type 'pip install -e creativeAI' to install the package"""

import setuptools

with open("README.md") as f:
    long_description = f.read()

setuptools.setup(
    name='creativeAI',
    version='0.0.1',
    url='https://github.com/mHead/creativeAI',
    author='Marco Testa',
    author_email='marco_testa@icloud.com',
    description='creativeAI: from music to image',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    install_requires=['torch',
                      'torchaudio',
                      'torchvision',
                      'tensorboard',
                      'scikit-learn',
                      'pandas',
                      'matplotlib',
                      'Pillow',
                      'pydub',
                      'scipy'],
    python_requires='>=3'
)