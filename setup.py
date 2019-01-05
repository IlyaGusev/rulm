from setuptools import find_packages, setup

setup(
    name='rulm',
    packages=find_packages(),
    version='0.0.1',
    description='',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rulm',
    download_url='https://github.com/IlyaGusev/rulm/archive/0.0.1.tar.gz',
    keywords=['nlp', 'russian', 'language model'],
    install_requires=[
        'numpy>=1.11.3',
        'pygtrie>=2.2',
        'torch>=1.0.0',
        'overrides>=1.9',
        'allennlp>=0.8.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',

        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: Russian',

        'Programming Language :: Python :: 3.6',
    ],
)
