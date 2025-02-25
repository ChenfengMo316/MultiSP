from setuptools import Command, find_packages, setup

__lib_name__ = "MultiSP"
__lib_version__ = "1.0.1"
__description__ = 'Accurate spatial domain detection by integrating spatial multi-omics data using MultiSP'
__url__ = "https://github.com/ChenfengMo316/MultiSP"
__author__ = "Chenfeng Mo"
__author_email__ = "mochenfeng316@whu.edu.cn"
__license__ = "MIT"
__keywords__ = ["Spatial multi-omics", "Spatially multimodal heterogeneity","Spatial domains", "Deep learning"]
__requires__ = ["requests",]

'''with open("README.rst", "r", encoding="utf-8") as f:
    __long_description__ = f.read()'''

setup(
    name = __lib_name__,
    version = __lib_version__,
    description = __description__,
    url = __url__,
    author = __author__,
    author_email = __author_email__,
    license = __license__,
    packages = ["MultiSP"],
    install_requires = __requires__,
    zip_safe = False,
    include_package_data = True
)