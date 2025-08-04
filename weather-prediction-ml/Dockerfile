ARG BUILD_FROM
FROM $BUILD_FROM

# Set Python environment variables
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies for ML
RUN apk add --no-cache \
    python3 \
    py3-pip \
    gcc \
    g++ \
    python3-dev \
    musl-dev \
    linux-headers \
    gfortran \
    openblas-dev \
    lapack-dev \
    libgomp \
    cmake \
    jq \
    curl

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install ML dependencies step by step
RUN python3 -m pip install numpy==1.24.4
RUN python3 -m pip install pandas==2.0.3
RUN python3 -m pip install scikit-learn==1.3.0
RUN python3 -m pip install xgboost==1.7.6
RUN python3 -m pip install joblib==1.3.1

# Install MQTT and other dependencies
RUN python3 -m pip install \
    paho-mqtt==1.6.1 \
    ha-mqtt-discoverable==0.13.1 \
    requests==2.31.0 \
    python-dateutil==2.8.2

# Copy application files
COPY app /app
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Copy ML models
COPY models /models

# Set working directory
WORKDIR /app

CMD [ "/run.sh" ]