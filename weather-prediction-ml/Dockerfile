ARG BUILD_FROM
FROM $BUILD_FROM

# Install system dependencies
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

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install Python ML packages with specific versions
COPY requirements.txt /
RUN python3 -m pip install --no-cache-dir -r /requirements.txt

# Copy application files
COPY app /app
COPY run.sh /
RUN chmod a+x /run.sh

# Set working directory
WORKDIR /app

# Labels
LABEL \
    io.hass.name="Weather Prediction ML" \
    io.hass.description="Machine learning weather prediction addon" \
    io.hass.type="addon" \
    io.hass.version=${BUILD_VERSION}

CMD [ "/run.sh" ]