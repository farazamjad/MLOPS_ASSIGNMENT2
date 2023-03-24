# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container and install the required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Train the model and save it to disk
RUN python model.py

# Set the environment variable to run the Flask app
ENV FLASK_APP=app.py

# Expose the port that the app will be running on
EXPOSE 5000

# Run the command to start the Flask app when the container starts
CMD ["flask", "run", "--host=0.0.0.0"]