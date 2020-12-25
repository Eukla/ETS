import setuptools

with open("README.md", "r") as fh:
    long_text = fh.read()

setuptools.setup(
    name="ets",
    version="0.1",
    author="Evgenios Kladis, Evangelos Michelioudakis",
    description="A library of early time-series classification algorithms.",
    long_description=long_text,
    long_description_content_type="text/markdown",
    license='GPL3',
    packages=setuptools.find_packages(),
    entry_points='''
        [console_scripts]
        ets=ets.cli:cli
    ''',
    classifiers=[
        "Programming Language :: Python :: 3",
        "GNU General Public License v3': 'License :: OSI Approved :: GNU General Public License v3 (GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
