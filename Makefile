NOSETESTS ?= nosetests

all:
	python setup.py build_ext -i

coverage:
	nosetests --with-coverage --cover-html --cover-package=sirl sirl

test-code: all
	$(NOSETESTS) -s -v sirl


clean:
	find | grep .pyc | xargs rm

test-doc:
	$(NOSETESTS) -s -v doc/*.rst doc/modules/

test: test-code test-doc
