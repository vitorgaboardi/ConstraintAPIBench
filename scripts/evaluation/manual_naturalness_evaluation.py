"""
Selects n random samples from the generated dataset and evaluates their naturalness by outputting 'yes' or 'no' for each sample. 
"""

import random
import json
import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score

path = "/home/vitor/Documents/phd/ConstraintAPIBench/results/dataset_quality_evaluation"
models = ["deepseek-v3", "gpt-4o"]
methods = ["constraint-aware", "sheng"]
judges = ["gpt-4.1", "Meta-Llama-3.1-70B-Instruct-Turbo"]
number_of_samples = 100
random.seed(50)
gpt_kappa_scores = []
llama_kappa_scores = []

for model in models:
    for method in methods:
        output_filepath = f"{path}/{model}/{method}/manual_naturalness_evaluation.csv"

        # check if the output file already exists to avoid overwriting existing evaluations
        if not os.path.exists(output_filepath):
            same_samples = 0
            results = []

            # retrieving for gpt-4.1 judge
            judge = "gpt-4.1"
            filepath = f"{path}/{model}/{method}/naturalness_by_{judge}_final.csv"
            gpt_naturalness_results = pd.read_csv(filepath)
            indexes = random.sample(range(len(gpt_naturalness_results)), number_of_samples)

            # retrieving for Meta-Llama-3.1-70B-Instruct-Turbo judge
            judge = "Meta-Llama-3.1-70B-Instruct-Turbo"
            filepath = f"{path}/{model}/{method}/naturalness_by_{judge}_final.csv"
            meta_naturalness_results = pd.read_csv(filepath)

            for index in indexes:
                utterance_gpt = gpt_naturalness_results.loc[index, "utterance"]
                
                sample = {
                    "api": gpt_naturalness_results.loc[index, "api"],
                    "api_method": gpt_naturalness_results.loc[index, "api_method"],
                    "utterance": gpt_naturalness_results.loc[index, "utterance"],
                    "human_judge": None,
                    "gpt-4.1_judge": gpt_naturalness_results.loc[index, "evaluation"],
                    "Meta-Llama-3.1-70B-Instruct-Turbo_judge": meta_naturalness_results.loc[index, "evaluation"],
                    
                }
                results.append(sample)
        
            pd.DataFrame(results).to_csv(output_filepath, index=False)

        # check if the output file already exists to avoid overwriting existing evaluations
        else: 
            results = pd.read_csv(output_filepath)

            # counting how many samples were judged as natural by the human judge
            natural_samples = results[results["human_judge"] == "natural"]
            
            print(f"Evaluation for {model} - {method}.")
            print(f"Number of samples judged as natural: {len(natural_samples)} out of {len(results)}")

            # computing inter-agreement between the human judge and the gpt-4.1 judge
            human_labels = results["human_judge"].apply(lambda x: 1 if x == "natural" else 0)
            gpt_labels = results["gpt-4.1_judge"].apply(lambda x: 1 if x == "natural" else 0)
            llama_labels = results["Meta-Llama-3.1-70B-Instruct-Turbo_judge"].apply(lambda x: 1 if x == "natural" else 0)
            kappa_gpt = cohen_kappa_score(human_labels, gpt_labels)
            kappa_llama = cohen_kappa_score(human_labels, llama_labels)
            gpt_kappa_scores.append(kappa_gpt)
            llama_kappa_scores.append(kappa_llama)
            print(f"Cohen's kappa between human judge and gpt-4.1 judge: {kappa_gpt}")
            print(f"Cohen's kappa between human judge and Meta-Llama-3.1-70B-Instruct-Turbo judge: {kappa_llama}")
            print("="*100)


print(f"Average Cohen's kappa between human judge and gpt-4.1 judge: {sum(gpt_kappa_scores)/len(gpt_kappa_scores)}")
print(f"Standard deviation of Cohen's kappa between human judge and gpt-4.1 judge: {pd.Series(gpt_kappa_scores).std()}")
print(f"Average Cohen's kappa between human judge and Meta-Llama-3.1-70B-Instruct-Turbo judge: {sum(llama_kappa_scores)/len(llama_kappa_scores)}")
print(f"Standard deviation of Cohen's kappa between human judge and Meta-Llama-3.1-70B-Instruct-Turbo judge: {pd.Series(llama_kappa_scores).std()}")