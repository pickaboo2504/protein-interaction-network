build:
	python setup.py sdist bdist_wheel

clean:
	rm -rf dist/*

upload:
	twine upload dist/*

release: clean build upload
