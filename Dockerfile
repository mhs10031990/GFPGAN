# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster
RUN pip install flask
RUN apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/gradient-ai/GFPGAN.git
WORKDIR /GFPGAN
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install opencv-python-headless
RUN pip install realesrgan==0.2.2.4
RUN pip install basicsr
RUN pip install facexlib
RUN python setup.py develop
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth

EXPOSE 5000

COPY . .

ENTRYPOINT [ "python" ]

CMD [ "app.py" ]