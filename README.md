# Constraint-Aware Pipeline for Generating High-Quality Utterances from OpenAPI Specifications using LLMs

# Installation Instructions

To set up the environment, please follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ConstraintAPIBench.git
   cd ConstraintAPIBench
   ```
2. Create a conda virtual environment and activate it:
   ```bash
    conda create -n constraints python=3.8 -y
    conda activate constraints
    ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

# Scripts

The repository contains scripts to run several key components. 
These scripts read information from configuration files that can be found in the `configs/` directory to define parameters and settings for each component.
Here are the main folders with scripts for different purposes

- `data_generation`: Generates data based on OpenAPI specifications using different prompting methods.
- `evaluation`: Evaluates the generated data using various metrics and methods.
- `preprocessing`: Preprocesses the data for testing or fine-tuning embedding models.

## Data generation

To generate data, navigate to the `data_generation` folder and run the desired script. 
The file `config_gen_data.yaml` contains the configuration parameters for data generation, such as the LLM to use, temperature, number of utterances, output folder for saving the generated data, and the path for the OpenAPI specifications.

Example command to run data generation:
```bash
python cap_generation.py
```

Currently, the path `/data/dataset` contains the generated dataset using two LLMs and two prompting methods.

## Testing dataset


## Pre-processing


## API retrieval training



## Evaluation

### Dataset Quality


### API Retrieval Performance