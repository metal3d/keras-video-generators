build:
	python3 setup.py sdist


.ONESHELL:
doc:
	source v/bin/activate
	cd sphinx-docs
	make html
	cp -ra _build/html ../docs

clean:
	rm -rf *.egg-info build dist
