from setuptools import find_packages, setup

VERSION = '0.0.1'
DESCRIPTION = 'SwAPP FL Group Project:Fednas test script'
LONG_DESCRIPTION = 'SwAPP FL Group Project:Fednas test script'

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="fedlib",
    version=VERSION,
    author="Sixing",
    author_email="yusx@iastate.edu",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    license="BSD-2-Clause License",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=['scikit-learn',
                    'pyyaml',
                    'ipykernel',
                    'transformers',
                    'tensorboard',
                    ],
    keywords=['python', 'dataset package', 'model package'],
    classifiers=[
        "Development Status :: Pre ALpha",
        "Intended Audience :: Education",
        "Intended Audience :: FAIR Research",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ]
)
