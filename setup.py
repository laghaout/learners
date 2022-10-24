from setuptools import find_packages, setup

setup(
    name='learners',
    version='0.0.1',
    author='Amine Laghaout',
    description='Object-oriented learners for machine learning.',
    packages=find_packages(exclude=('test*',)),
    zip_safe=False,
    entry_points={
        'console_scripts': ['learners-mock = learners.main:main', ]}
)
