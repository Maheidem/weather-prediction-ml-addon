ARG BUILD_FROM
FROM $BUILD_FROM

# Install only essential dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    curl \
    jq

# Install minimal Python packages
RUN python3 -m pip install --no-cache-dir \
    paho-mqtt==1.6.1 \
    ha-mqtt-discoverable==0.13.1 \
    requests==2.31.0

# Copy application files
COPY app_simple /app
COPY run_simple.sh /run.sh
RUN chmod a+x /run.sh

# Set working directory
WORKDIR /app

CMD [ "/run.sh" ]