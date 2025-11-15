"""
Constraint extractor using LLMs to identify parameter constraints from API specifications.
"""

import json
import re
import copy
import os
from typing import Dict, List
from tqdm import tqdm
from openai import OpenAI
from .prompts import CONSTRAINT_EXTRACTION, EXAMPLE_INPUT_CONSTRAINT, EXAMPLE_OUTPUT_CONSTRAINT

class ConstraintExtractor:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)

        # Base LLM prompt
        self.constraint_extraction_prompt = [
            {"role": "system", "content": CONSTRAINT_EXTRACTION},
            {"role": "user", "content": EXAMPLE_INPUT_CONSTRAINT},
            {"role": "assistant", "content": EXAMPLE_OUTPUT_CONSTRAINT},
        ]

    def extract_constraints(self, api_path: str, temperature: float = 0.0) -> Dict:
        """Extract constraints for all methods in a single API specification file."""
        with open(api_path, 'r') as f:
            data = json.load(f)

        api_name = data.get('tool_name', '')
        api_description = data.get('tool_description', '')
        api_url = data.get('home_url', '')

        print(f"Processing API: {api_name}")

        api_methods_to_save = []
        for api_method in data.get('api_list', []):
            api_method_name = api_method.get('name', '')
            api_method_description = api_method.get('description', '')
            api_method_parameters = api_method.get('parameters', [])
            api_method_url = api_method.get('url', '')

            # print(f"   └── Method: {api_method_name}")

            # Only call LLM if there are parameters
            if len(api_method_parameters) > 0:
                input_data = {
                    "API Name": api_name,
                    "API Description": api_description,
                    "API Method Name": api_method_name,
                    "API Method Description": api_method_description[:4000],
                    "Parameters": api_method_parameters,
                }

                messages = copy.deepcopy(self.constraint_extraction_prompt) + [{"role": "user", "content": str(input_data)}]
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1000,
                    temperature=temperature
                )

                # Attach parsed constraints to parameters
                api_method_parameters = self._process_llm_output(response, api_method_parameters)

            api_methods_to_save.append({
                "name": api_method_name,
                "description": api_method_description,
                "url": api_method_url,
                "parameters": api_method_parameters
            })

        documentation = {
            "name": api_name,
            "description": api_description,
            "url": api_url,
            "api_methods": api_methods_to_save
        }

        return documentation

    def _process_llm_output(self, llm_response, api_method_parameters: List[Dict]) -> List[Dict]:
        """Parse and attach constraints from the LLM output to the parameter list."""
        try:
            content = llm_response.choices[0].message.content
            content = re.sub(r"```(json)?", "", content).strip()
            content = re.sub(r"//.*", "", content)
            constraints = json.loads(content)

            for parameter in api_method_parameters:
                name = parameter.get("name")
                if name in constraints:
                    parameter['constraints'] = constraints[name]

        except Exception as e:
            print("Error parsing LLM output when extracting constraints.")

        return api_method_parameters
