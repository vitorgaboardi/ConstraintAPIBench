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
from evaluation.metrics import naturalness_evaluation, bertscore, cosine_similarity, parameter_coverage, parameter_combination_coverage, constraint_adherance
from sklearn.metrics import cohen_kappa_score

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
    cfg = load_config(Path(__file__).parent.parent.parent / "config" / "config_quality_evaluation.yaml")
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
    evaluate_constraint_adherance = cfg.get("evaluation", {}).get("constraint_adherance", False)
    evaluate_cohen_kappa = cfg.get("evaluation", {}).get("cohen_kappa", False)

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
    print(f"These are the oas: {oas_to_evaluate}")

    # 1 - evaluating naturalness
    if evaluate_naturalness:
        print("Evaluating Naturalness...")
        detailed_results = []
        summarised_results = []

        for llm, url, api_key in zip(llm_as_judge_name, llm_url, api_keys):
            output_folder = Path(__file__).parent.parent.parent / "results" / "dataset_quality_evaluation" / llm_name / prompt_to_evaluate
            os.makedirs(output_folder, exist_ok=True)            
            
            rows = []
            natural_count = 0
            unnatural_count = 0
            wrong_count = 0

            for category_index, filename in enumerate(tqdm(oas_to_evaluate, desc="APIs")):
                if category_index >= 0:
                    file_path = os.path.join(utterances_path, filename)
                    with open(file_path, "r") as f:
                        oas = json.load(f)
                    print(f"{category_index} - Evaluating filename: {filename}")
                    
                    results_naturalness = naturalness_evaluation(oas=oas, api_key=api_key, base_url=url, model_name=llm)
                    natural_count += results_naturalness['natural_count']
                    unnatural_count += results_naturalness['unnatural_count']
                    wrong_count += results_naturalness['wrong_count']
                    
                    # extend rows with detailed results
                    rows.extend(results_naturalness["detailed_results"])

                    # save after each file
                    output_file = output_folder / f"naturalness_by_{llm.split('/')[-1]}.csv"
                    df = pd.DataFrame(rows)
                    df.to_csv(output_file, index=False)
                    print(f"💾 Saved partial results to {output_file}")
            
            # Summarized results for the current LLM
            summarised_results.append({
                "llm_as_judge": llm,
                "natural_count": natural_count,
                "unnatural_count": unnatural_count,
                "wrong_count": wrong_count,
                "total_utterances": natural_count + unnatural_count + wrong_count,
            })

            # ✅ Final save of all detailed results for this LLM
            final_output_file = output_folder / f"naturalness_by_{llm.split('/')[-1]}_final.csv"
            pd.DataFrame(rows).to_csv(final_output_file, index=False)
            print(f"✅ Saved detailed naturalness results to {final_output_file}")

        # ✅ Save summarized results across all LLMs
        summary_output = output_folder / "naturalness_summary.csv"
        pd.DataFrame(summarised_results).to_csv(summary_output, index=False)
        print(f"✅ Saved summarised naturalness results to {summary_output}")

    if evaluate_cohen_kappa:
        print("Evaluating Cohen's Kappa between LLM Judges...")
        output_folder = Path(__file__).parent.parent.parent / "results" / "dataset_quality_evaluation" / llm_name / prompt_to_evaluate
        all_judges_results = {}

        for llm in llm_as_judge_name:
            input_file = output_folder / f"naturalness_by_{llm.split('/')[-1]}_final.csv"
            df = pd.read_csv(input_file)
            df["evaluation"] = df["evaluation"].map({"natural": 1, "unnatural": 0, "wrong": -1})
            all_judges_results[llm] = df["evaluation"].tolist()

        # Compute Cohen's Kappa for each pair of judges
        kappa_results = []
        for (llm1, results1), (llm2, results2) in itertools.combinations(all_judges_results.items(), 2):
            kappa_score = cohen_kappa_score(results1, results2)
            kappa_results.append({
                "judge_1": llm1,
                "judge_2": llm2,
                "cohen_kappa": kappa_score
            })
            print(f"Cohen's Kappa between {llm1} and {llm2}: {kappa_score:.4f}")
        # Save Cohen's Kappa results
        kappa_output_file = output_folder / "cohen_kappa_results.csv"
        pd.DataFrame(kappa_results).to_csv(kappa_output_file, index=False)
        print(f"✅ Saved Cohen's Kappa results to {kappa_output_file}")

    # 3 - evaluating parameter diversity
    if evaluate_parameter_diversity:
        print("Evaluating Parameter Diversity...")
        parameter_coverage_value = []
        parameter_combination_coverage_value = []
        total_parameters = 0
        APIs_evaluated = 0
        for category_index, filename in enumerate(tqdm(oas_to_evaluate, desc="APIs")):
            file_path = os.path.join(utterances_path, filename)  # path to the API spec file
            with open(file_path, "r") as f:
                oas = json.load(f)
            print(f"Evaluating the following API: {filename}")

            # computing parameter coverage
            pc, num_params = parameter_coverage(oas)
            if pc is not None:
                parameter_coverage_value.append(pc)
                total_parameters += num_params

            # computing parameter combination coverage
            pcc, _ = parameter_combination_coverage(oas)
            if pcc is not None:
                parameter_combination_coverage_value.append(pcc)

            APIs_evaluated += 1
        
        average_pc = round(sum(parameter_coverage_value) / len(parameter_coverage_value), 4)
        pcc = sum(parameter_combination_coverage_value)
        print(f"Total number of parameters across evaluated APIs: {total_parameters}")
        print(f"Number of APIs evaluated for Parameter Coverage: {APIs_evaluated}")
        print(f"Average parameter coverage (PC) across evaluated APIs: {average_pc}")
        print(f"Total parameter combination coverage (PCC) across evaluated APIs: {pcc}")

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
            cs = cosine_similarity(oas, embedding_model=embedding_model)
            cosine_similarity_scores.append(cs)

            # computing BERTScore
            bs = bertscore(oas, embedding_model=embedding_model_bs)
            bertscore_scores.append(bs)

        average_cs = round(sum(cosine_similarity_scores) / len(cosine_similarity_scores), 4)
        average_bs = round(sum(bertscore_scores) / len(bertscore_scores), 4)
        print(f"Average Semantic Relevance across evaluated APIs: {average_cs}")
        print(f"Average BERTScore across evaluated APIs: {average_bs}")

    if evaluate_constraint_adherance:
        print("Evaluating Constraint Adherance...")
        constraint_gt_folder = cfg["constraint_gt_folder"]
        constraint_violations_list = [0, 0, 0]

        for category_index, filename in enumerate(tqdm(oas_to_evaluate, desc="APIs")):
            file_path = os.path.join(utterances_path, filename)  # path to the API spec file
            with open(file_path, "r") as f:
                oas = json.load(f)
            print(f"Evaluating the following API: {filename}")

            # computing constraint adherance
            violations = constraint_adherance(oas, ground_truth_path=constraint_gt_folder + filename)
            constraint_violations_list[0] += violations[0]
            constraint_violations_list[1] += violations[1]
            constraint_violations_list[2] += violations[2]

        total_violations = sum(constraint_violations_list)
        print(f"Total Max/Min Constraint Violations across evaluated APIs: {constraint_violations_list[0]}")
        print(f"Total Format Constraint Violations across evaluated APIs: {constraint_violations_list[1]}")
        print(f"Total Inter-dependency Constraint Violations across evaluated APIs: {constraint_violations_list[2]}")
        print(f"Total Constraint Violations across evaluated APIs: {total_violations}")


if __name__ == '__main__':
    main()