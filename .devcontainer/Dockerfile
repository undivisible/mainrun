FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    git \
    jq \
    zip \
    build-essential \
    ca-certificates \
    sudo \
    && apt-get clean

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get update && apt-get install -y nodejs

RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

RUN npm install -g zx

COPY requirements.txt* requirements-train.txt* /tmp/
RUN if [ -f /tmp/requirements.txt ]; then pip3 install --no-cache-dir -r /tmp/requirements.txt; fi
RUN if [ -f /tmp/requirements-train.txt ]; then pip3 install --no-cache-dir -r /tmp/requirements-train.txt; fi

RUN echo "devcontainer" >> /root/.mainrun