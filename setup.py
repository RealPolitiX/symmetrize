import setuptools

setuptools.setup(
    name="symmetrize",
    version="0.1.0",
    url="https://github.com/RealPolitiX/symmetrize",

    author="R. Patrick Xian",
    author_email="xrpatrick@gmail.com",

    description="Symmetrization and centering of 2D pattern using nonrigid point set registration",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
