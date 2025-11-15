import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
from typing import Dict, List
from openai import OpenAI
from .prompts import PROMPT_UTTERANCE_GENERATION


class UtteranceGenerator:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_utterances(self, oas: Dict, num_utterances: int = 10, temperature: float = 0.3) -> List[str]:
        """Generate constraint-aware utterances for a given API method."""            
        api_name = oas['name']
        api_description = oas['description']

        # iterating through all API methods
        for api_method_index, api_method in enumerate(oas['api_methods']):
            api_method_name = api_method['name']
            api_method_description = api_method['description'] 
            api_method_parameters = [param for param in api_method['parameters'] if not ('constraints' in param and param['constraints'].get('technical') is True)]

            # defining required and optional parameters
            required_parameters = [param["name"] for param in api_method_parameters if param.get('required', False)]
            optional_parameters = [param["name"] for param in api_method_parameters if 'required' in param and param['required'] is False]

            # preparing input for LLM
            api_specification = {"API Name": api_name,
                                "API Description": api_description if len(api_description) < 4000 else '',
                                "API Method Name": api_method_name,
                                "API Method Description": api_method_description if len(api_method_description) < 4000 else '',
                                "Parameters": api_method_parameters}

            if len(required_parameters) + len(optional_parameters) == 0:
                input = (f"API Specification: \n{api_specification}\n"
                         f"Write {num_utterances} utterances that use the given API.\n")
            elif len(optional_parameters) == 0:
                input = (f"API Specification: \n{api_specification}\n"
                            f"Write {num_utterances} utterances that use the given API.\n"
                            f"Required parameters: {str(required_parameters).strip('[]')}.\n")
            elif len(required_parameters) == 0:
                input = (f"API Specification: \n{api_specification}\n"
                            f"Write {num_utterances} utterances that use the given API.\n"
                            f"Optional parameters: {str(optional_parameters).strip('[]')}.\n")
            else:
                input = (f"API Specification: \n{api_specification}\n"
                            f"Write {num_utterances} utterances that use the given API.\n"
                            f"Required parameters: {str(required_parameters).strip('[]')}.\n"
                            f"Optional parameters: {str(optional_parameters).strip('[]')}.\n")
            
            # prompt definition and calling LLM
            messages = [{"role": "system", "content": PROMPT_UTTERANCE_GENERATION},
                        {"role": "user", "content": str(input)}]
            response = self.openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=3000,
                        temperature=temperature)
            api_method['utterances'] = self._process_llm_output(response)
            print(f"   └── Generated {len(api_method['utterances'])} utterances for method: {api_method_name}")
        return oas

    def _process_llm_output(self, llm_response) -> List[Dict]:
        """Process LLM output to extract and attach constraints to parameters."""
        try: 
            content = llm_response.choices[0].message.content
            content = content.replace("```python","").replace("```","")
            content = content.replace("True", "true").replace("False", "false").replace("None", "null")
            content = json.loads(content)
            return content

        except Exception as e:
            print("Error parsing LLM output when generating utterances.")
            return 'error parsing the information'
