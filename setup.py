import os, shutil


# Move tutorials inside mercury.monitoring before packaging
if os.path.exists('tutorials'):
    shutil.move('tutorials', 'mercury/monitoring/tutorials')


from setuptools import setup, find_packages


setup_args = dict(
	name				 = 'mercury-monitoring',
	packages			 = find_packages(include = ['mercury*', 'tutorials*']),
	include_package_data = True,
	package_data		 = {'mypackage': ['tutorials/*', 'tutorials/data/*']}
)

setup(**setup_args)
