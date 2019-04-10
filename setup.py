from setuptools import find_packages, setup
import os
import codecs

from conveiro import __version__

with codecs.open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
    description = f.read()


setup(
    author='The ShowmaxLab & Showmax teams',
    author_email='oss+conveiro@showmax.com',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        "Topic :: Scientific/Engineering",
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    description='Visualization of filters in convolutional neural networks',
    include_package_data=True,
    install_requires=['numpy', 'matplotlib', 'scipy', 'pillow'],
    extras_require = {
        'gpu' : ['tensorflow-gpu'],
        'cpu' : ['tensorflow'],
        'examples' : ['tensornets'],
        'cli': ['tensornets', 'click', 'graphviz']
    },
    entry_points = {
        'console_scripts': [
            'conveiro = conveiro.cli:run_app'
        ]
    },
    keywords=['CNN', 'neural networks', 'deep dream', 'visualization'],
    license='Apache License, Version 2.0',
    long_description=description,
    long_description_content_type='text/markdown',
    name='conveiro',
    packages=find_packages(),
    python_requires = "~=3.4",
    platforms=['any'],
    version=__version__,
    url='https://github.com/showmax/conveiro',
)
