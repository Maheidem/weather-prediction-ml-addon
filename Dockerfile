ARG BUILD_FROM
FROM $BUILD_FROM

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    py3-numpy \
    py3-scipy \
    gcc \
    g++ \
    python3-dev \
    musl-dev \
    linux-headers \
    gfortran \
    openblas-dev \
    lapack-dev \
    libgomp \
    jq \
    curl

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python ML packages
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt

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