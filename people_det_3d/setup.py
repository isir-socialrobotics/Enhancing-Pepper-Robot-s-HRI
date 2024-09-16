from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# Fetch values from package.xml
setup_args = generate_distutils_setup(
    packages=['people_det_3d'],  # This should be the name of your package directory
    package_dir={'': 'src'},     # The location of your Python package inside the `src` folder
)

setup(**setup_args)
