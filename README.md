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
These scripts read information from configuration files that can be found in the `config/` directory to define parameters and settings for each component. The main folders with scripts are:

- `data_generation`: Generates synthetic utterances from OpenAPI specifications using different prompting methods (e.g., CAP and baselines)
- `evaluation`: Evaluates both dataset quality and retrieval performance using the scripts in scripts/evaluation/.
- `preprocessing`: Transforms raw generated data into the retrieval training/testing format.

## Data generation

To generate data, navigate to the `data_generation` folder and run the desired script. 
The file `config/config_gen_data.yaml` contains the configuration parameters for data generation, such as the LLM to use, temperature, number of utterances, output folder for saving the generated data, and the path for the OpenAPI specifications. Users can edit these parameters as needed.

Example command to run data generation using CAP:
```bash
python scripts/data_generation/cap_generation.py
```

Currently, the path `/data/dataset` contains the generated dataset using two LLMs and two prompting methods.

## Testing dataset

The test dataset is derived from [ToolRet](https://huggingface.co/datasets/mangopy/ToolRet-Queries), which standardizes user queries from different benchmarks and dataset.

We used testing dataset provided by Gorilla, ToolAlpaca, MetaTool, and T-eval, leading to a total of 942 utterances across 633 APIs. 

Use the following command to generate the testing dataset:

```bash
python scripts/preprocessing/retrieval_dataset_test.py
```

## Pre-processing

It is necessary to organize and preprocess the generated dataset before training embedding models. 
The file `config/config_retriever_dataset_preprocess.yaml` contains the configuration parameters for preprocessing the training dataset, such as the input folder with the generated data, the output folder for saving the preprocessed data, the definition of the prompt used to generated the data, and the LLM used to generate the data.

Command to run preprocessing for training dataset:

```bash
python scripts/preprocessing/retrieval_dataset_train.py
```

## API retrieval training

Once that the training and testing datasets are preprocessed in the correct format, you can proceed to train the API retrieval model.
The file `config/config_retriever_training.yaml` contains the configuration parameters for training the API retrieval model, such as (1) training dataset path, (2) testing dataset path, (3) output folder for saving the trained model, (4) prompt designed used to generate the data, (5) LLM used to generate the data, and (6) training hyperparameters (embedding model, epochs, train_batch_size, learning_rate, warmup steps, and max_seq_length).

Command to run training for API retrieval model after defining the parameters in the configuration file:

```bash
python scripts/evaluation/retrieval_train.py
```

Currently, this process will run five trainings with different seeds for each configuration to ensure the robustness of the results, saving each model in the output folder defined in the configuration file.
The code for training is based on the code available from the [ToolBench](https://github.com/OpenBMB/ToolBench/tree/master/toolbench/retrieval) repository. 

## Evaluation

We evaluate the performance of the trained API retrieval models in terms of the quality of the dataset and the API retrieval performance. 

### Dataset Quality

We evaluate the quality of the dataset in three dimensions: (1) naturalness, (2) parameter diversity, and (3) constraint adherance. Each dimension has different metrics to assess the quality of the generated utterances. The metric definitions and computation can be found under `src/evaluation/metrics/`.
The file `config/config_quality_evaluation.yaml` contains the configuration parameters for evaluating the dataset quality, such as (1) input folder with the generated utterance dataset, (2) the folder with the ground truth constraint definitions (in case of constraint adherence evaluation), (3) the prompt to evaluate, (4) the LLM used to generate the data to evaluate, (5) the number of APIs to evaluate, and (6) the random seed. Other configurations related to the metrics can be found in the configuration file.

Command to run dataset quality evaluation:

```bash
python scripts/evaluation/dataset_quality_evaluation.py
```

### API Retrieval Performance

It is also possible to evaluate any pre-trained or fine-tuned model in the testing dataset. 
The path of the testing dataset and the model to evaluate can be defined inside the python file to run the evaluation.

Command to run API retrieval evaluation:

```bash 
python scripts/evaluation/retrieval_evaluation.py
```