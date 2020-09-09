from setuptools import setup

url = ""
version = "0.0.1"
readme = open('README.rst').read()

setup(
    name="camps",
    packages=["camps"],
    version=version,
    description="not really camping",
    long_description=readme,
    include_package_data=True,
    author="Adam Schnapp",
    author_email="adschnapp@gmail.com",
    url=url,
    download_url="{}/tarball/{}".format(url, version),
    license="MIT"
)
