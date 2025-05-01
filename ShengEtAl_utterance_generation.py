# recreating the generation of utterance using the prompt from Sheng et Al: Measuring an LLM’s Proficiency at using APIs: A Query Generation Strategy

import json
import ast
import copy
import os
import numpy as np
import random
import itertools
import re
from openai import OpenAI

PROMPT_INSTRUCTION = """
In this task, you are given a description of a tool, including it's parameters, generated from the tool's OpenAPI specification. Your task is to come up with questions in natural language form that can be answered by invoking the tool.

You must adhere to the following rules:
#1: For each question, list the parameters in the API that it populates, along with corresponding values.
#2: The parameter values in the question must be human readable.
#3: Please use the parameter values verbatim in the question.
#4: Use natural sounding values for the parameters. For example, do not use British pounds for American flights.
#5: Some API parameters are required fields, so every question must contain them.
#6: Do not create (or populate) new parameters. Only use existing parameters in the API.
#7: Try to cover all API parameters in the list of questions below.
#8: Generate a diverse set of questions: do not repeat the same question more than once.

The output must be a Python list of dictionaries, where each dictionary has two keys:
    - "utterance": the natural language request.
    - "parameters": a dictionary containing the name-value pairs for all parameters used in the utterance. 
    
The "parameters" dictionary must have pairs that can be infered or recognised from the generated natural utterance. 
Finally, you must only output the Python list and do not output anything else, such as notes or explanation about the reasoning.
"""

# they include one example in their prompt. check if it is fair to include as I am not providing any example in my own prompt. 
EXAMPLE_INPUT = """ Tool Specification:
{ 
info:
  title: google_flights_tool
  description: Google Flights tool to search and get booking links for flights.
  - url: http://flights.google.com
paths:
  /search:
    get:
      operationId: search
      description: Search or get booking links for flights.
      parameters:
        - name: origin
          description: The location where the trip starts. If the location is not specified in the user query, it should be left empty.
          type: string
          required: false
        - name: destination
          description: The final destination of the trip. If it is not provided in the user query, it should be left empty.
          type: string
          required: false
        - name: earliest_departure_date
          description: Filter for the earliest (or only) departure date in YYYY-MM-DD format.
          type: string
          required: false
        - name: earliest_return_date
          description: Filter for the earliest (or only) return date in YYYY-MM-DD format.
          type: string
          required: false
        - name: latest_departure_date
          description: Filter for the latest departure date in YYYY-MM-DD format. Set this to be the same as earliest_departure_date unless a departure date range is requested.
          type: string
          required: false
        - name: latest_return_date
          description: Filter for the latest return date in YYYY-MM-DD format. Set this to be the same as earliest_return_date unless a return date range is requested.
          type: string
          required: false
        - name: min_length_of_stay
          description: Minimum length of stay before returning.
          type: integer
          required: false
        - name: max_length_of_stay
          description: Maximum length of stay before returning.
          type: integer
          required: false
        - name: include_airlines
          description: Filter by flights on these airlines only.
          type: array of strings
          required: false
        - name: max_duration_minutes
          description: Filter for maximum duration of the flight.
          type: integer
          default: -1base_messages = [{"role": "system", "content": PROMPT_INSTRUCTION},
          required: false
        - name: depart_after_hour
          description: Filter for flights that depart after this hour (1-23).
          type: integer
          required: false
        - name: depart_before_hour
          description: Filter for flights that depart before this hour (1-23).
          type: integer
          required: false
        - name: arrive_after_hour
          description: Filter for flights that arrive after this hour (1-23).
          type: integer
          required: false
        - name: arrive_before_hour
          description: Filter for flights that arrive before this hour (1-23).
          type: integer
          required: false
        - name: carry_on_bag_count
          description: Filter for number of carry on bags.
          type: integer
          default: -1
          required: false
        - name: checked_bag_count
          description: Filter for number of checked bags.
          type: integer
          default: -1
          required: false
        - name: trip_days
          description: Filter for specific grouping of flight trips like Weekends.
          required: false
        - name: seating_classes
          description: Filter for seating classes of the flight.
          type: array of strings
          enum: [ECONOMY_CLASS,ECONOMY_PLUS_CLASS,BUSINESS_CLASS,FIRST_CLASS,SUITES_CLASS]
          required: false
        - name: cheapest
          description: If TRUE, the results will be sorted by price (in ascending order).
          type: boolean
          required: false
        - name: one_way
          description: If TRUE, a query will be issued for a one-way trip instead of round-trip (the default).
          type: boolean
          required: false
        - name: num_adult_passengers
          description: The number of adult passengers.
          type: integer
          required: false
        - name: num_child_passengers
          description: The number of child passengers.
          type: integer
          required: false
        - name: num_infant_in_lap_passengers
          description: The number of infant passengers (in lap).
          type: integer
          required: false
        - name: num_infant_in_seat_passengers
          description: The number of infant passengers (in seat).
          type: integer
          required: false
        - name: pet_friendly
          type: boolean
}

Write 7 utterances that use the specified tool. 
# Required parameters: 'destination'. So, destination must be present in every question.
"""

EXAMPLE_OUTPUT = """
[
    {
    "utterance": "book flights from SF to NYC for a 3 day business trip.",
    "parameters": {
        "origin": "SF",
        "destination": "NYC"}
    },
    {
    "utterance": "round trip-ticket from DC to london in premium economy leaving on Aug 22 and returning on Sep 3rd.",
    "parameters": {
        "origin": "DC",
        "destination": "london",
        "earliest_departure_date": "2023-08-22",
        "earliest_return_date": "2023-09-03",
        "seating_classes": "premium economy",
        "one_way": false}
    },
    {
    "utterance": "united flight for 2 adults and 1 lap infant from LAX for a 3 day trip to Portland starting Sep 7th.",
    "parameters": {
        "origin": "LAX",
        "destination": "Portland",
        "earliest_departure_date": "2023-09-07",
        "earliest_return_date": "2023-09-10",
        "num_adult_passengers": 2,
        "num_infant_in_lap_passengers": 1,
        "include_airlines": "united"}
    },
    {
    "utterance": "search for flights from hong kong to sydney nonstop.",
    "parameters": {
        "origin": "hong kong",
        "destination": "sydney",
        "max_stops": 0}
    },
    {
    "utterance": "Find round trip suites class flights from zurich to japan for a trip starting Aug 29 and returning on Sep 3rd.",
    "parameters": {
        "origin": "zurich",
        "destination": "japan",
        "earliest_departure_date": "2023-08-29",
        "earliest_return_date": "2023-09-03",
        "seating_classes": "suites",
        "one_way": false}
    },
    {
    "utterance": "search for a premium or economy class flight from vancouver to honolulu leaving on Dec 14th and returning on Dec 17th with at most one stop.",
    "parameters": {
        "origin": "vancouver",
        "destination": "honolulu",
        "earliest_departure_date": "2023-12-14",
        "earliest_return_date": "2023-12-17",
        "max_stops": 1,
        "seating_classes": ["economy class", "premium economy"]}
    },
    {
    "utterance": "get economy class booking options for 2 adults and 1 child from Miami to Seattle leaving on Feb 20th and returning on Feb 23rd for under $1000.",
    "parameters": {
        "origin": "Miami",
        "destination": "Seattle",
        "earliest_departure_date": "2024-02-20",
        "earliest_return_date": "2024-02-23",
        "seating_classes": "economy class",
        "num_adult_passengers": 2,
        "num_child_passengers": 1,
        "max_price": 1000,
        "currency": "USD"}
    }
]
"""

# models
#model="gpt-4.1-mini"
#model="gpt-4.1"
model='deepseek-ai/DeepSeek-V3'


# defining paths
OAS_folder = './dataset/tools'

if model == "gpt-4.1-mini" or model == "gpt-4.1":
  client = OpenAI(api_key="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")
  utterance_folder = './dataset/Sheng et al prompt/'+model+'/utterances/'
else:
  client = client = OpenAI(api_key="0cTfrP7S4xnxVcQeMnkCvfY7nrgZs41e",
                           base_url="https://api.deepinfra.com/v1/openai",)
  model_name = model.split('/')[1]
  utterance_folder = './dataset/Sheng et al prompt/'+model_name+'/utterances/'

# adding examples or not:
number_of_utterances_per_method = 10
include_example = True
API_methods_count = 0
API_count = 0
mistakes = 0

OAS_to_create_utterances = ['foreca_weather.json-Daily', 'learn_to_read_and_write_japanese_kanji.json-Kanji grade level', 'getitsms_whatsapp_apis.json-GetIT SMS WHATSAPP API', 'streaming_availability.json-Search Basic (FREE)', 'solarenergyprediction.json-/v2.0/solar/prediction', 'nowpayments.json-3.Getestimatedprice', 'referential.json-Languages', 'covid_19_by_api_ninjas.json-/v1/covid19', 'ott_details.json-Advanced Search', 'dezgo.json-/text2image', 'veriphone.json-verify', 'working_days.json-/1.3/list_non_working_days', 'flightera_flight_data.json-airportDelayDailyStatistics', 'postal_ninja.json-createTrack', 'cricket_live_data.json-Results By Date', 'car_code.json-/obd2/{code}', 'trackingpackage.json-TrackingPackage', 'shazam.json-songs/get-count', 'axesso_amazon_data_service.json-lookupSeller', 'hotels_com_provider.json-Hotel Rooms (offers)', 'netflix_v2.json-Search', 'recipe_food_nutrition.json-Generate Shopping List', 'spotify.json-Artist albums', 'movie_database_alternative.json-By Search', 'tasty.json-tips/list', 'synwave.json-Upload a new file', 'everyearthquake.json-Earthquakes', 'flightera_flight_data.json-airlineStatistics', 'webcams_travel.json-/webcams/list/webcam={webcamid}[,{webcamid}[,...]]', 'working_days.json-/1.3/add_working_days']

if include_example:
    base_messages = [{"role": "system", "content": PROMPT_INSTRUCTION},
                    {"role": "user", "content": EXAMPLE_INPUT},
                    {"role": "assistant", "content": EXAMPLE_OUTPUT}]
else:
    base_messages = [{"role": "system", "content": PROMPT_INSTRUCTION}]

categories = sorted(os.listdir(OAS_folder))
for category_index, category in enumerate(categories):
    category_path = os.path.join(OAS_folder, category)
    #print(category_path)

    # generate folder to save OAS now enriched with utterances
    saving_utterances_path = os.path.join(utterance_folder, category)

    # saving file with utterances and parameters mapping
    if not os.path.exists(saving_utterances_path):
        os.makedirs(saving_utterances_path)

    # iterating over the OAS for each category
    for root, _, files in os.walk(category_path):
        for filename in files:
            # make sure that the file is not there
            if not os.path.exists(saving_utterances_path+'/'+filename): 
                # read file for each API
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    api_name = data['tool_name']
                    api_description = data['tool_description']

                    #print(category_index, category, data['tool_name'])
                    call_made = False

                    # iterating over each API method
                    for api_method_index, api_method in enumerate(data['api_list']):
                        # this is temporary, just to generate utterances for the APIs that will be used!
                        api_method_name = api_method['name']
                        if filename+"-"+api_method_name in OAS_to_create_utterances:
                            print(API_methods_count, filename+"-"+api_method_name)
                            api_method_description = api_method['description']
                            api_method_parameters = api_method['parameters']
                            required_parameters = [param["name"] for param in api_method_parameters if param.get('required', False)]

                            print('api_method:', api_method_name)

                            # all the parameters will be used! even the ones that were detected after the 'api_related' because that is related to the constraints part only that I decided to remove!
                            tool_specification = {"API Name": api_name,
                                                "API Description": api_description if len(api_description) < 4000 else '',
                                                "API Method Name": api_method_name,
                                                "API Method Description": api_method_description if len(api_method_description) < 4000 else '',
                                                "Parameters": api_method_parameters}
                            
                            if required_parameters:
                                input = (
                                    f"Tool Specification: \n{tool_specification}\n"
                                    f"Write {number_of_utterances_per_method} utterances that use the specified tool.\n"
                                    f"# Required parameters: {str(required_parameters).strip('[]')}."
                                    f" So, {str(required_parameters).strip('[]')} must be present in every utterance."
                                )
                            else:
                                input = (
                                    f"Tool Specification: \n{tool_specification}\n"
                                    f"Write {number_of_utterances_per_method} utterances that use the specified tool."
                                )

                            messages = base_messages + [{"role": "user", "content": str(input)}]

                            # calling API
                            call_made = True
                            response = client.chat.completions.create(
                                model=model,
                                messages=messages,
                                max_tokens=3000,
                                temperature=0)

                            try: 
                                content = response.choices[0].message.content
                                content = content.replace("```python","").replace("```","")
                                content = content.replace("True", "true").replace("False", "false").replace("None", "null")
                                content = json.loads(content)

                                api_method['utterances'] = content

                            except Exception as e:
                                print("Exception arised!")
                                print(e)
                                print(response.choices[0].message.content)
                                api_method['utterances'] = 'error parsing the information.'
                                mistakes+=1

                            API_methods_count+=1

                # this is just for now!
                if call_made:
                    with open(saving_utterances_path+'/'+filename, 'w') as json_file:
                        json.dump(data, json_file, indent=4) 
                #print()
            
            API_count+=1

print('number of methods:', API_methods_count)
print('mistakes:', mistakes)
print('APIs:', API_count)