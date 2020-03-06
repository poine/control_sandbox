clean:
	find . -name '*~' -exec rm {} \;
	find . -name '*.pyc' -exec rm {} \;
	find . -name '__pycache__' -exec rm -rf {} \;


test_doc:
	cd docs; bundle exec jekyll serve
