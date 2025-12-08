from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="pure-visual-grounder",
    version="1.0.2",
    author="Strategion",
    author_email="development@strategion.de",
    description="A package for processing PDFs with vision-based language models",
    long_description="This package uses the given LLM and pdf to perform the OCR operation. Technical documents are "
                     "often in need to be stored in RAG and lack the uniform structure. This package helps you to get "
                     "the relevant data out of the pdf",
    long_description_content_type="text/markdown",
    packages=find_packages(include=['pure_visual_grounding', 'pure_visual_grounding.*']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)