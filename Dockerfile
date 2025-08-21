FROM ghdl/ghdl:6.0.0-dev-gcc-ubuntu-22.04
LABEL maintainer=sioni@cern.ch
RUN apt update && \
    apt install -y build-essential wget git ca-certificates && \
    useradd --create-home --shell /bin/bash conifer
# Add Tini
ENV TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT ["/tini", "--"]
USER conifer
ENV WORKDIR=/home/conifer
WORKDIR $WORKDIR
COPY . .
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3.sh -b -p "${HOME}/conda" && \
    source "${HOME}/conda/etc/profile.d/mamba.sh" && \
    mamba activate && \
    mamba shell init && \
    pip install -r dev_requirements.txt && \
    git clone --depth 1 --branch v3.12.0 https://github.com/nlohmann/json.git && \
    git clone --depth 1 https://github.com/Xilinx/HLS_arbitrary_Precision_Types.git && \
    pip install .
ENV JSON_ROOT=${WORKDIR}/json/single_include
ENV XILINX_AP_INCLUDE=${WORKDIR}/HLS_arbitrary_Precision_Types/include
ENV PATH="${WORKDIR}/conda/bin:${PATH}"
CMD ["/bin/bash"]