from setuptools import setup

setup(
    name='learners',
    version='dev',
    author='Amine Laghaout',
    description='Object-oriented learners for machine learning.',
    install_requires=[
        'seaborn>=0.11.0',
        'matplotlib>=3.3.2',
        'numpy>=1.19.2',
        'pandas>=1.1.3'
        'scikit-learn>=0.23.2',
        'tensorflow>=2.4.1'
    ],
    packages=['learners'],
    python_requires=">=3.8",
    zip_safe=False,
)
