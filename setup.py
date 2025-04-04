from setuptools import setup, find_packages


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return f.read().splitlines()


setup(
    name='ae',
    version='0.1.6',
    packages=find_packages(include=["ae", "ae.*"]),
    exclude_package_data={"examples": ["examples/*"], "flows": ["flows/*"]},
    url='https://github.com/shill1729/research',
    install_requires=parse_requirements("requirements.txt"),
    license='MIT',
    author='Sean Hill',
    author_email='52792611+shill1729@users.noreply.github.com',
    description='Regularized Autoencoders with Riemannian SDEs'
)
