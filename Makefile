setup:
	python3 -m venv mlops
		#source mlops/bin/activate
install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt