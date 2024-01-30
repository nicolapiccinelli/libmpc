FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /

##############################
# Core tools
##############################

RUN apt-get update -y -qq \
    && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        lsb-release \
        build-essential \
        software-properties-common \
        ca-certificates \
        gpg-agent \
        wget \
        git \
        cmake \
        lcov \
        gcc \
        clang \
        clang-tidy \
        clang-format \
        libomp-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


##############################
# Non-root user Setup
##############################
ARG USERNAME=dev
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        sudo \
        gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/$USERNAME --shell /bin/bash --user-group --groups adm,sudo $USERNAME \
    && echo $USERNAME:$USERNAME | chpasswd \
    && echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL >> /etc/sudoers \
    && touch home/$USERNAME/.sudo_as_admin_successful \
    && gosu $USERNAME mkdir -p /home/$USERNAME/.xdg_runtime_dir
ENV XDG_RUNTIME_DIR=/home/$USERNAME/.xdg_runtime_dir


##############################
# Eigen
##############################
RUN apt-get update -y -qq \
    && apt-get install -y -qq --no-install-recommends \
        libeigen3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


##############################
# NL Optimization
##############################
RUN git clone https://github.com/stevengj/nlopt /tmp/nlopt \
    && cd /tmp/nlopt \
    && mkdir build \
    && cd build \
    && cmake \
        -D CMAKE_BUILD_TYPE=Release \
        -D NLOPT_PYTHON=OFF \
        -D NLOPT_OCTAVE=OFF \
        -D NLOPT_MATLAB=OFF \
        -D NLOPT_GUILE=OFF \
        -D NLOPT_SWIG=OFF \
        .. \
    && make -j$(($(nproc)-1)) \
    && make install \
    && rm -rf /tmp/*


##############################
# OSQP Solver
##############################
RUN git clone --depth 1 --branch v0.6.3 --recursive https://github.com/osqp/osqp /tmp/osqp \
    && cd /tmp/osqp \
    && mkdir build \
    && cd build \
    && cmake \ 
        -G "Unix Makefiles" \
        .. \
    && make -j$(($(nproc)-1)) \
    && make install \
    && rm -rf /tmp/*


##############################
# Catch2
##############################
RUN git clone https://github.com/catchorg/Catch2.git /tmp/Catch2 \
    && cd /tmp/Catch2 \
    && mkdir build \
    && cd build \
    && cmake \ 
        -D BUILD_TESTING=OFF \
        .. \
    && make -j$(($(nproc)-1)) \
    && make install \
    && rm -rf /tmp/*

# # (Optional) Set Clang as the default compiler
# ENV CC=/usr/bin/clang
# ENV CXX=/usr/bin/clang++