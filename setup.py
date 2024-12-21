from setuptools import setup, find_packages

setup(
    name="LatentBoostClassifier",  # Package name
    version="1.2.3",               # Package version
    author="Ali Bavarchee",
    author_email="ali.bavarchee@gmail.com",
    description="A hybrid generative model combining CVAE, CGAN, and Random Forest.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AliBavarchee/LatentBoostClassifier",  # GitHub repo URL
    packages=find_packages(),      # Automatically detect package modules
    include_package_data=True,     # Include additional files specified in MANIFEST.in
    install_requires=[
        "tensorflow>=2.8.0",
        "keras-tuner>=1.1.0",
        "scikit-learn>=1.0",
        "matplotlib>=3.4",
        "seaborn>=0.11",
        "numpy>=1.19",
        "pandas>=1.1",
        "tqdm>=4.62"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires=">=3.7",
    keywords="machine learning generative models CVAE CGAN Random Forest",
    license="MIT",
    project_urls={
        "Documentation": "https://github.com/AliBavarchee/LatentBoostClassifier#readme",
        "Source": "https://github.com/AliBavarchee/LatentBoostClassifier",
        "Tracker": "https://github.com/AliBavarchee/LatentBoostClassifier/issues"
    },
)
