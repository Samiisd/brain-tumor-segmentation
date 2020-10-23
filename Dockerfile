FROM python:3.7

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN useradd -m audran
USER audran
WORKDIR /work

ENTRYPOINT jupyter notebook --NotebookApp.ip=0.0.0.0
