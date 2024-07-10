from setuptools import setup, find_packages

# get description from readme file
with open('README.md', 'r') as f:
    long_description = f.read()

# setup
setup(
    name='HyperbolicEmbeddings',
    version='0.1',
    description='',
    long_description = long_description,
    long_description_content_type="text/markdown",
    author='Noémie Jaquier, Leonel Rozo',
    author_email='noemie.jaquier@kit.edu, leonel.rozo@de.bosch.com',
    maintainer=' ',
    maintainer_email='',
    license=' ',
    url=' ',
    platforms=['Linux Ubuntu'],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
)
