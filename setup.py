import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="NeuralNetworkPY",
    version="0.0.1",
    author="Andrew Farley",
    author_email="amf7crazy@gmail.com",
    description="An easy-to-use Neural Network library I'm building from scratch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/And1210/NeuralNetworkPY",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)