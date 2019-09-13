FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04
#FROM ubuntu:16.04

RUN apt update && apt install -y build-essential libbz2-dev libdb-dev libreadline-dev libffi-dev libgdbm-dev \
liblzma-dev libncursesw5-dev libsqlite3-dev libssl-dev  zlib1g-dev uuid-dev tk-dev wget unzip libglu1-mesa-dev \
libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg sudo

RUN wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz && tar xzf Python-3.6.9.tgz && rm Python-3.6.9.tgz
RUN cd Python-3.6.9 && ./configure && make -j14 && make install 

ENV USER hond
ENV HOME /home/${USER}

ADD ./requirements.txt ${HOME}/requirements.txt
RUN pip3 install -U setuptools pip
RUN pip3 install -r ${HOME}/requirements.txt

RUN echo 'root:pass_r' |chpasswd
RUN useradd -m ${USER} &&  gpasswd -a ${USER} sudo && echo "${USER}:pass_h" | chpasswd
WORKDIR ${HOME}
USER ${USER}

CMD ["/bin/bash"]