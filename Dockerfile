FROM nvidia/cuda:9.0-cudnn7-devel

# Install miniconda3
ENV MINICONDA_HOME="/opt/miniconda"
ENV PATH="${MINICONDA_HOME}/bin:${PATH}"
RUN apt update &&\
    DEBIAN_FRONTEND=noninteractive apt install -y curl &&\
    curl -o /tmp/Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    chmod +x /tmp/Miniconda3-latest-Linux-x86_64.sh &&\
    /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}" &&\
    rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Install requirements
COPY environment.yml /tmp/environment.yml
RUN conda env update -n=root --file=/tmp/environment.yml
RUN conda clean -y -a &&\
    rm /tmp/environment.yml
