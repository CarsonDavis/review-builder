from setuptools import find_packages, setup


# Function to parse requirements.txt
def parse_requirements(filename):
    with open(filename) as file:
        lines = file.read().splitlines()
        # Filter out empty lines and comments
        requirements = [line for line in lines if line and not line.startswith("#")]
    return requirements


setup(
    name="book_summarizer",
    version="0.1",
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),
)
