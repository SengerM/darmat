import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="darmat",
	version="0.0.0",
	author="Matias H. Senger",
	author_email="m.senger@hotmail.com",
	description="DarMat Experiment package",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/SengerM/darmat",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"Operating System :: Ubuntu 18.04",
	],
	package_data = {
		# ~ '': ['rc_styles/*']
	}
)
