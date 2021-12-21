# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/gradient-ai/GFPGAN.git
WORKDIR /GFPGAN
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install realesrgan==0.2.2.4
RUN pip install basicsr
RUN pip install facexlib
RUN python setup.py develop

COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]