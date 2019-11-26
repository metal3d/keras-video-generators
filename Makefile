build:
	python3 setup.py sdist


.ONESHELL:
doc:
	source v/bin/activate
	cd docs
	make html

clean:
	rm -rf *.egg-info build dist
