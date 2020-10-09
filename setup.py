import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ipypostgis',
    version='0.0.1',
    author='Joona Laine',
    author_email='joona@gispo.fi',
    description='PostGIS utils for Jupyter Notebook',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GispoCoding/ipypostgis',
    packages=setuptools.find_packages(),
    install_requires=[
        'geopandas>=0.8.1',
        'ipython-sql>=0.4.0',
        'keplergl>=0.2.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Framework :: Jupyter',
        'Topic :: Database',
        'Development Status :: 4 - Beta',
    ],
    python_requires='>=3.6',
)
