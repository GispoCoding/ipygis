import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='ipygis',
    version='0.0.1',
    author='Joona Laine',
    author_email='joona@gispo.fi',
    description='GIS utils for Jupyter Notebook',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/GispoCoding/ipygis',
    packages=setuptools.find_packages(),
    install_requires=[
        'geoalchemy2>=0.9.0',
        'geopandas>=0.8.1',
        'h3>=3.7.3',
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
