import setuptools
import atexit
# imports for doing post-installation things
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import check_call

def _post_install():
    """
    Post-installation command to run after install.
    """
    post_install_cmds = [
    "jupyter labextension install jupyterlab_bokeh",
    "jupyter serverextension enable --py nbserverproxy"]
    for cmd in post_install_cmds:   
        check_call(cmd.split())

class new_install(install):
    def __init__(self, *args, **kwargs):
        super(new_install, self).__init__(*args, **kwargs)
        atexit.register(_post_install)

with open("README.md", "r") as fh:
    long_description = fh.read()

# read in the requirements
requirements_path = 'requirements.txt'
try:
    with open(requirements_path) as f:
        requirements = [line for line in f.read().splitlines() if len(line) > 0]
except FileNotFoundError:
    print('No requirments.txt file found! Install will fail.')

setuptools.setup(

     name='mlviz',  

     version='0.1',

     install_requires=requirements,

     author="F. P. Chmiel & Z. D. Zlako",

     author_email="fpc@it-innovation@soton.ac.uk",

     description="A interactive visualisation package to support exploratory"
                 "analysis of high dimensional data.",

     long_description=long_description,

     include_package_data=True,

     cmdclass={
               'install': new_install,
              },

     long_description_content_type="text/markdown",

     url="None",

     packages=setuptools.find_packages(),

     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: NOT PUBLICALLY RELEASED",
         "Operating System :: OS Independent",
     ],
 )