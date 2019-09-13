FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

RUN apt update && apt install -y build-essential libbz2-dev libdb-dev libreadline-dev libffi-dev libgdbm-dev liblzma-dev libncursesw5-dev libsqlite3-dev libssl-dev  zlib1g-dev uuid-dev tk-dev wget unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg

RUN wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz && tar xzf Python-3.6.9.tgz
RUN cd Python-3.6.9 && bash configure && \
make -j14 && make install 

ADD ./src /home/root/src
ADD ./requirements.txt /home/root/requirements.txt

RUN pip3 install -U setuptools pip
RUN pip3 install -r /home/root/requirements.txt
