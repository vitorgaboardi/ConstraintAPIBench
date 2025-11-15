""""
Script to evaluate the quality of a generated dataset.
Does for a single prompt/pipeline and LLM model.
"""

import os
import sys
import json
import yaml
import random
import itertools
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
from evaluation.metrics import naturalness_evaluation, bertscore, cosine_similarity, parameter_coverage, parameter_combination_coverage

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def load_config(path: Path) -> dict:
    """Loads configuration to be used in the generation method."""
    if not path.exists():
        sys.exit(1)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    required = ["llm_to_evaluate", "prompt_to_evaluate", "number_of_apis_to_evaluate"]
    for key in required:
        if key not in cfg:
            print(f"[red]Error:[/] Missing required field '{key}' in {path}")
            sys.exit(1)
    return cfg

def main():
    # loading config information
    cfg = load_config(Path(__file__).parent.parent / "config" / "config_quality_evaluation.yaml")
    llm_name = cfg["llm_to_evaluate"]
    utterances_path = cfg["utterances_folder"]
    prompt_to_evaluate = cfg["prompt_to_evaluate"]
    random_seed = cfg["random_seed"]
    number_of_apis_to_evaluate = cfg["number_of_apis_to_evaluate"]
    embedding_model_cs = cfg["embedding_model"]["cosine_similarity"]
    embedding_model_bs = cfg["embedding_model"]["bertscore"]

    # evaluation flags
    evaluate_naturalness = cfg.get("evaluation", {}).get("naturalness", False)
    evaluate_parameter_diversity = cfg.get("evaluation", {}).get("parameter_diversity", False)
    evaluate_semantic_relevance = cfg.get("evaluation", {}).get("semantic_relevance", False)

    # defining the LLMs as judges
    llm_as_judge_name = cfg["llm_as_judge"]["name"]
    llm_url = cfg["llm_as_judge"]["url"]
    llm_temp = cfg["llm_as_judge"]["temperature"]
    api_keys = cfg["llm_as_judge"]["api_key"]

    # defining the API methods to evaluate
    random.seed(random_seed)
    utterances_path = os.path.join(utterances_path, llm_name, prompt_to_evaluate, "utterances")
    oas_to_evaluate = sorted(os.listdir(utterances_path))
    oas_to_evaluate = random.sample(oas_to_evaluate, min(number_of_apis_to_evaluate, len(oas_to_evaluate)))
    print(f"Evaluating {len(oas_to_evaluate)} APIs located in {utterances_path}.")

    # 1 - evaluating naturalness
    if evaluate_naturalness:
        print("Evaluating Naturalness...")
        detailed_results = []
        summarised_results = []

        for llm, url, api_key in zip(llm_as_judge_name, llm_url, api_keys):
            rows = []
            natural_count = 0
            unnatural_count = 0
            wrong_count = 0

            for category_index, filename in enumerate(tqdm(oas_to_evaluate, desc="APIs")):
                file_path = os.path.join(utterances_path, filename)  # path to the API spec file
                with open(file_path, "r") as f:
                    oas = json.load(f)
                
                results_naturalness = naturalness_evaluation(oas=oas, api_key=api_key, base_url=url, model_name=llm)

                natural_count += results_naturalness['natural_count']
                unnatural_count += results_naturalness['unnatural_count']
                wrong_count += results_naturalness['wrong_count']   
                rows.append(results_naturalness["detailed_results"])

                break
            
            summarised_results.append({
                "llm_as_judge": llm,
                "natural_count": natural_count,
                "unnatural_count": unnatural_count,
                "wrong_count": wrong_count,
                "total_utterances": natural_count + unnatural_count + wrong_count})
            
            detailed_results.extend(list(itertools.chain.from_iterable(rows)))
            
            # saving detailed results per model
            output_folder = Path(__file__).parent.parent / "results" / "dataset_quality_evaluation"
            os.makedirs(output_folder, exist_ok=True)
            output_file = output_folder / f"naturalness_{llm_name}_{prompt_to_evaluate}_by_{llm.split('/')[-1]}.csv"
            df = pd.DataFrame(detailed_results)
            df.to_csv(output_file, index=False)
            print(f"✅ Saved detailed naturalness results to {output_file}")

        # saving summarised results
        output_folder = Path(__file__).parent.parent / "results" / "dataset_quality_evaluation"
        os.makedirs(output_folder, exist_ok=True)
        output_file = output_folder / f"naturalness_{llm_name}_{prompt_to_evaluate}_summary.csv"
        df = pd.DataFrame(summarised_results)
        df.to_csv(output_file, index=False)
        print(f"✅ Saved summarised naturalness results to {output_file}")

    # 2 - evaluating parameter diversity
    if evaluate_parameter_diversity:
        print("Evaluating Parameter Diversity...")
        parameter_coverage_value = []
        parameter_combination_coverage_value = []
        APIs_evaluated = 0
        for category_index, filename in enumerate(tqdm(oas_to_evaluate, desc="APIs")):
            file_path = os.path.join(utterances_path, filename)  # path to the API spec file
            with open(file_path, "r") as f:
                oas = json.load(f)
            print(f"Evaluating the following API: {filename}")

            # computing parameter coverage
            pc = parameter_coverage(oas)
            if pc is not None:
                parameter_coverage_value.append(pc)

            # computing parameter combination coverage
            pcc = parameter_combination_coverage(oas)
            if pcc is not None:
                parameter_combination_coverage_value.append(pcc)

            APIs_evaluated += 1
        
        average_pc = round(sum(parameter_coverage_value) / len(parameter_coverage_value), 4)
        pcc = sum(parameter_combination_coverage_value)
        print(f"Number of APIs evaluated for Parameter Coverage: {APIs_evaluated}")
        print(f"Average Parameter Coverage across evaluated APIs: {average_pc}")
        print(f"Average Parameter Combination Coverage across evaluated APIs: {pcc}")

    # 3 - evaluating semantic relevance
    if evaluate_semantic_relevance:
        print("Evaluating Semantic Relevance...")
        cosine_similarity_scores = []
        bertscore_scores = []
        embedding_model = SentenceTransformer(embedding_model_cs)

        for category_index, filename in enumerate(tqdm(oas_to_evaluate, desc="APIs")):
            file_path = os.path.join(utterances_path, filename)  # path to the API spec file
            with open(file_path, "r") as f:
                oas = json.load(f)
            print(f"Evaluating the following API: {filename}")

            # computing cosine similarity
            # cs = cosine_similarity(oas, embedding_model=embedding_model)
            # cosine_similarity_scores.append(cs)

            # computing BERTScore
            bs = bertscore(oas, embedding_model=embedding_model_bs)
            bertscore_scores.append(bs)

        # average_cs = round(sum(cosine_similarity_scores) / len(cosine_similarity_scores), 4)
        average_bs = round(sum(bertscore_scores) / len(bertscore_scores), 4)
        # print(f"Average Semantic Relevance across evaluated APIs: {average_cs}")
        print(f"Average BERTScore across evaluated APIs: {average_bs}")

if __name__ == '__main__':
    main()