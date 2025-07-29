
FROM python:3.10-slim


WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY download_models.py .
RUN python download_models.py

# Set the environment variable for the cache in the final running container
ENV TRANSFORMERS_CACHE="/persistent_cache"
ENV SENTENCE_TRANSFORMERS_HOME="/persistent_cache"
# --- End of crucial part ---

# Copy the rest of your application code
COPY . .

# Command to run your main application when the container starts
CMD ["python", "your_main_script.py"]
