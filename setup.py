from setuptools import setup, find_packages

# how to release:
# UPDATE VERSION IN 3 PLACES: Ais/core/config.py, setup.py, docs/conf.py

# push to pypi:
# python setup.py sdist
# twine upload dist/*

setup(
    name='Pom',
    version='0.0.1',
    packages=find_packages(),
    entry_points={'console_scripts': ['pom=Pom.main:main']},
    license='GPL v3',
    author='mgflast',
    author_email='m.g.f.last@lumc.nl',
    long_description_content_type="text/markdown",
    package_data={'': ['*.png', '*.glsl', '*.pdf', '*.txt', '*.json']},
    include_package_data=False,  # weirdly, the above filetypes _are_ included when this parameter is set to False.
    install_requires=[
        "Ais-cryoET>=1.0.40",
        "matplotlib",
        "openpyxl",
        "pandas",
        "streamlit",
        "streamlit-aggrid"
    ]
)

