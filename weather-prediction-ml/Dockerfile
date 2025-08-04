ARG BUILD_FROM
FROM $BUILD_FROM

# Set Python environment variables
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Install system packages including pre-built ML libraries
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-scipy \
    py3-scikit-learn \
    py3-pandas \
    jq \
    curl

# Install remaining Python packages via pip
RUN python3 -m pip install \
    xgboost \
    joblib \
    paho-mqtt==1.6.1 \
    ha-mqtt-discoverable==0.13.1 \
    requests==2.31.0

# Copy application files
COPY app /app
COPY run.sh /run.sh
RUN chmod a+x /run.sh

# Copy ML models
COPY models /models

# Set working directory
WORKDIR /app

CMD [ "/run.sh" ]