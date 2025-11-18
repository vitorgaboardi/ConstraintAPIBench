import os
import sys
import json
import yaml
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from data_generation.cap.constraint_extractor import ConstraintExtractor
from data_generation.cap.utterance_generator import UtteranceGenerator

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def load_config(path: Path) -> dict:
    """Loads configuration to be used in the generation method."""
    if not path.exists():
        console.log(f"[red]Error:[/] Could not find configuration file at {path}")
        sys.exit(1)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    required = ["llm_name", "llm_url", "oas_path", "output_folder"]
    for key in required:
        if key not in cfg:
            console.log(f"[red]Error:[/] Missing required field '{key}' in {path}")
            sys.exit(1)
    return cfg


def main():
    # 1 - loading config information
    cfg = load_config(Path(__file__).parent.parent / "config" / "config_gen_data.yaml")
    llm_name = cfg["llm_name"]
    llm_url = cfg["llm_url"]
    llm_temp = cfg.get("llm_temp", 1.0)
    utterances = cfg.get("utterances", 10)
    oas_path = cfg["oas_path"]
    output_folder = Path(cfg["output_folder"], (llm_name.split('/')[-1]).lower(), "constraint-aware")
    api_key = os.getenv("OPENAI_API_KEY")

    # 2 - initializing extractor and utterance generator
    extractor = ConstraintExtractor(api_key=api_key, base_url=llm_url,model_name=llm_name)
    utterance_generator = UtteranceGenerator(api_key=api_key, base_url=llm_url, model_name=llm_name)

    # 3 - iterating through all OAS files
    categories = sorted(os.listdir(oas_path))
    for category_index, category in enumerate(tqdm(categories, desc="Categories")):
        category_path = os.path.join(oas_path, category)

        for root, _, files in os.walk(category_path):
            for filename in files:
                if not os.path.exists(os.path.join(output_folder, filename)):
                    file_path = os.path.join(category_path, filename)  # path to the API spec file 
                    
                    # 1 - check whether the constraints file already exists
                    if not os.path.exists(os.path.join(output_folder, "constraints", filename)):
                        # extracting and saving constraints
                        constraints = extractor.extract_constraints(file_path, llm_temp)
                        os.makedirs(os.path.join(output_folder, "constraints"), exist_ok=True)  # create output directory if not exists 
                        output_file = os.path.join(output_folder, "constraints" , filename)
                        print(f"Output file: {output_file}")
                        with open(output_file, "w") as f:
                            json.dump(constraints, f, indent=4)
                        print(f"✅ Saved extracted constraints of {filename}")
                    else:
                        with open(os.path.join(output_folder, "constraints", filename), "r") as f:
                            constraints = json.load(f)
                        print(f"The file {filename} already has extracted constraints")

                    # 2 - check whether the constraints file already exists
                    if not os.path.exists(os.path.join(output_folder, "utterances", filename)):
                        # generating constraint-aware utterances                    
                        oas_with_utterances = utterance_generator.generate_utterances(constraints, num_utterances=utterances, temperature=llm_temp)
                        os.makedirs(os.path.join(output_folder, "utterances"), exist_ok=True)  # create output directory if not exists 
                        output_file = os.path.join(output_folder, "utterances" , filename)
                        with open(output_file, "w") as f:
                            json.dump(oas_with_utterances, f, indent=4)
                        print(f"✅ Saved OAS with utterances of {filename}")
                    else:
                        print(f"The file {filename} already has utterances")

if __name__ == '__main__':
    main()