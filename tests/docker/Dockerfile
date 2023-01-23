FROM ubuntu:22.04

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y build-essential wget git ghdl
# Add Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod a+x /tini

RUN useradd --create-home --uid 1000 --shell /bin/bash conifer
USER conifer
WORKDIR /home/conifer

ENV PATH="/home/conifer/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /home/conifer/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    conda init bash && \
    . /home/conifer/.bashrc && \
    conda update conda

COPY environment.yml /tmp/environment.yml
RUN conda env create --file=/tmp/environment.yml
RUN git clone --depth 1 --branch v3.11.2 https://github.com/nlohmann/json && \
    git clone https://github.com/Xilinx/HLS_arbitrary_Precision_Types
ENV JSON_ROOT=/home/conifer/json/include
ENV XILINX_AP_INCLUDE=/home/conifer/HLS_arbitrary_Precision_Types/include
ENTRYPOINT ["/tini", "-s", "--"]
CMD [ "/bin/bash" ]