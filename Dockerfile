FROM ubuntu:focal

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /


##############################
# Core tools
##############################

RUN apt-get update -y -qq \
    && apt-get install -y -qq --no-install-recommends \
        apt-utils \
        lsb-release \
        ca-certificates \
        apt-transport-https \
        software-properties-common \
        build-essential \
        pkg-config \
        wget \
        git \
        cmake \
        gpg-agent \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


##############################
# Non-root user Setup
##############################
RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        sudo \
        gosu \
        locales \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*
ENV LANG en_US.UTF-8

# Create a (non-root) user
ARG USERNAME=dev
RUN useradd --create-home --home-dir /home/$USERNAME --shell /bin/bash --user-group --groups adm,video,sudo,dialout $USERNAME \
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
RUN git clone --recursive https://github.com/osqp/osqp /tmp/osqp \
    && cd /tmp/osqp \
    && mkdir build \
    && cd build \
    && cmake \ 
        -G "Unix Makefiles" \
        .. \
    && cmake --build . \
    && cmake --build . --target install \
    && rm -rf /tmp/*


##############################
# Clang
##############################
ARG CLANG_VERSION="15"
RUN wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && bash llvm.sh ${CLANG_VERSION} all \
    && ln /usr/bin/clang-${CLANG_VERSION} -f /usr/bin/clang \
    && ln /usr/bin/clang++-${CLANG_VERSION} -f /usr/bin/clang++ \
    && ln /usr/bin/clang-tidy-${CLANG_VERSION} -f /usr/bin/clang-tidy \
    && ln /usr/bin/clang-format-${CLANG_VERSION} -f /usr/bin/clang-format \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Clang as the default compiler
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++