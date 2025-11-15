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
from .prompts import PROMPT_UTTERANCE_GENERATION, EXAMPLE_INPUT, EXAMPLE_OUTPUT

class UtteranceGenerator:
    def __init__(self, api_key: str, base_url: str = None, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.openai_client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_messages = [{"role": "system", "content": PROMPT_UTTERANCE_GENERATION},
                              {"role": "user", "content": EXAMPLE_INPUT},
                              {"role": "assistant", "content": EXAMPLE_OUTPUT}]

    def generate_utterances(self, oas: Dict, num_utterances: int = 10, temperature: float = 0.3) -> List[str]:
        """Generate constraint-aware utterances for a given API method."""            
        api_name = oas.get('tool_name', '')
        api_description = oas.get('tool_description', '')
        home_url = oas.get('home_url', '')

        API_methods_to_save = []
        for api_method_index, api_method in enumerate(oas['api_list']):
            # getting API method information
            api_method_name = api_method['name']
            api_method_description = api_method['description']
            api_method_url = api_method['url']
            api_method_parameters = api_method['parameters']
            required_parameters = [param["name"] for param in api_method_parameters if param.get('required', False)]

            api_specification = {"API Name": api_name,
                                 "API Description": api_description if len(api_description) < 4000 else '',
                                 "API Method Name": api_method_name,
                                 "API Method Description": api_method_description if len(api_method_description) < 4000 else '',
                                 "Parameters": api_method_parameters}

            # preparing input for LLM
            if required_parameters:
                input = (f"Tool Specification: \n{api_specification}\n"
                         f"Write {num_utterances} utterances that use the specified tool.\n"
                         f"# Required parameters: {str(required_parameters).strip('[]')}."
                         f" So, {str(required_parameters).strip('[]')} must be present in every utterance.")
            else:
                input = (f"Tool Specification: \n{api_specification}\n"
                         f"Write {num_utterances} utterances that use the specified tool.")

            messages = self.base_messages + [{"role": "user", "content": str(input)}]
            response = self.openai_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=3000,
                        temperature=temperature)  
            api_method['utterances'] = self._process_llm_output(response)
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