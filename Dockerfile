FROM python:3.10-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install CPU-only torch to save space and download time
# We need torch>=2.6.0 to fix the CVE-2025-32434 vulnerability, but PyTorch 2.6.0+ CPU wheels
# haven't been published to the main index yet or have dependency issues.
# Instead, we install the latest torch 2.6+ from the default PyPI index, but force CPU-only.
RUN pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu

# Install requirements
COPY --chown=user ./requirements.txt /app/requirements.txt
# Remove torch from requirements.txt to avoid reinstalling the heavy CUDA version
RUN sed -i '/^torch$/d' /app/requirements.txt && \
    pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application
COPY --chown=user . /app

# Command to run the FastAPI application on port 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
