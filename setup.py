from setuptools import setup, find_packages

setup(
    name='nceigsd',
    version='0.1.0',
    description='A library for processing Global Summary of the Day (GSD) data from NCEI-NOAA',
    author='Quang-Van Doan',
    author_email='doan.van.gb@...',
    url='https://github.com/yourusername/nceigsd',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "wget",
        "matplotlib"
    ],
    extras_require={
        "plot": ["seaborn", "basemap"]
    },
    package_data={
        "nceigsd": ["data/list-isd-history_2024.csv"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.7',
)