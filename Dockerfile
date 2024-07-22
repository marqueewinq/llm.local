# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the application
COPY ./llmlocal /app/
ENV PYTHONPATH=$PYTHONPATH:/app/
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8001"]
