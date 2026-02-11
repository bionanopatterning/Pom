from setuptools import setup, find_packages

# how to release:
# UPDATE VERSION IN 3 PLACES: Ais/core/config.py, setup.py, docs/conf.py

# push to pypi:
# python setup.py sdist
# twine upload dist/*

setup(
    name='Pom-cryoET',
    version='1.2.0',
    packages=find_packages(),
    entry_points={'console_scripts': ['pom=Pom.main:main']},
    license='GPL v3',
    author='Mart G. F. Last',
    author_email='mlast@mrc-lmb.cam.ac.uk',
    long_description_content_type="text/markdown",
    package_data={
        '': ['*.png', '*.glsl', '*.pdf', '*.txt', '*.json'],
        'Pom.core': ['defaults/*.json']
    },
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "pandas",
        "streamlit>=1.50.0",
        "streamlit-aggrid",
        "starfile",
        "tqdm",
    ]
)

