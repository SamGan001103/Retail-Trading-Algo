from setuptools import find_packages, setup

setup(
    name="retail-trading-algo",
    version="0.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=["python-dotenv", "requests", "signalrcore"],
)
