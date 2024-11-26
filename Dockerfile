# start with official anaconda base image
FROM continuumio/anaconda3:latest

# set working directory in container
WORKDIR /app

# copy environment.yml file to container
COPY environment.yml /app/environment.yml

# install dependencies from environment.yml file
RUN conda env create -f /app/environment.yml

# set default shell to use created conda environment
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]

# install additional tools if necessary 
RUN conda install -n base -c conda-forge <additional-packages>

# copy application code to container
COPY . /app

# copy pickle model into container
COPY final_xgboost_model.pkl /app/final_xgboost_model.pkl

# expose application port
EXPOSE 8000

# Define the command to run your app (adjust to your entry point)
CMD ["conda", "run", "-n", "base", "python", "predict.py"]
