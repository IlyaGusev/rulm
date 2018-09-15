from setuptools import find_packages, setup

setup(
    name='rulm',
    packages=find_packages(),
    version='0.0.0',
    description='',
    author='Ilya Gusev',
    author_email='phoenixilya@gmail.com',
    url='https://github.com/IlyaGusev/rulm',
    download_url='https://github.com/IlyaGusev/rulmh/archive/0.0.0.tar.gz',
    keywords=['nlp', 'russian', 'language model'],
    install_requires=[
        'numpy>=1.11.3',
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',

        'Topic :: Text Processing :: Linguistic',

        'License :: OSI Approved :: Apache Software License',

        'Natural Language :: Russian',

        'Programming Language :: Python :: 3.6',
    ],
)
