# Use an official Python runtime as a base image
FROM python:3.9.16-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install the necessary Python libraries
# Replace with the libraries you need for your machine learning project
RUN pip install numpy pandas scikit-learn torch datetime tqdm joblib PyWavelets

# You can set a default command or entry point, such as opening a shell
CMD ["/bin/bash"]

