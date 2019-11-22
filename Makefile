build:
	python3 setup.py bdist_wheel sdist


clean:
	rm -rf *.egg-info build dist
