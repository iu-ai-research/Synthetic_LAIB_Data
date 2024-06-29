from openai import OpenAI
import anthropic
import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access your API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# messages = [
#     { 
#       "role": "system", 
#       "content": "You are an expert in synthetic data generation for laser-accelerated ion beams. When prompted with original samples you will generate 50 new synthetic rows from the whole original dataset using your broad domain knowledge and deep understanding for the original features, the generated synthetic samples should be realistic and representative of the original data including the target_material frequency. Respond with synthetic rows in a CSV file format."
#     },
#     { "role": "user", 
#       "content": '''
#         intensity	pulse_width	cutoff_energy	power	spot_size	target_material	target_thickness
#         2.38E+21	106.19	500	20.2	2.12E+14	2.8044649	Gold (Au)	10
#         ...
#         '''
#     }
# ]

def openai_req(
        model:str,
        # max_tokens:int,
        messages:list[dict]
    ):
    client = OpenAI(api_key=OPENAI_API_KEY)

    # ChatCompletion(
    #     id='chatcmpl-9adlPGrnqBBnydsJzuefVy8NXIMPl', 
    #     choices=[
    #         Choice(
    #             finish_reason='stop', 
    #             index=0, 
    #             logprobs=None, 
    #             message=ChatCompletionMessage(content="### Step-by-Step Generation of Data Points\n\n#### Step 1: Analyze Patterns ... or domain-specific requirements.", role='assistant', function_call=None, tool_calls=None)
    #         )
    #     ], 
    #     created=1718519687, 
    #     model='gpt-4o-2024-05-13', 
    #     object='chat.completion', 
    #     system_fingerprint='fp_f4e629d0a5', 
    #     usage=CompletionUsage(completion_tokens=2983, prompt_tokens=1515, total_tokens=4498))

    api_response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    response = {
        'content': api_response.choices[0].message.content,
        'api_obj': api_response
    }

    return response

def anthropic_req(
        model:str,
        max_tokens:int,
        messages:list[dict]
    ):

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Message(
    #     id='msg_013VimtzTPBZ979Mi2cCBMWc', 
    #     content=[TextBlock(text='Here is a step-by-step approach to synthesizing 25 .... median and percentiles.', type='text')], 
    #     model='claude-3-opus-20240229', 
    #     role='assistant', 
    #     stop_reason='end_turn', 
    #     stop_sequence=None, 
    #     type='message', 
    #     usage=Usage(input_tokens=1630, output_tokens=1467)
    # )

    api_response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=messages[0]['content'], # <-- system prompt as specific argument for anthropic api requests
        messages=[
            messages[1]
        ]
    )

    response = {
        'content': api_response.content[0].text,
        'api_obj': api_response
    }

    return response

# For making requests to a local ollama server -- ollama version is 0.1.38
def ollama_req(
        model:str,
        # max_tokens:int,
        messages:list[dict],
        #seed:int=0
    ):

    OLLAMA_SERVER = 'http://localhost:11434/api/generate'

    api_response = requests.post(OLLAMA_SERVER,
                  timeout=60*60,
                  data = json.dumps(
                    {
                        "model": f"{model}",
                        "system":messages[0]['content'],
                        "prompt":messages[1]['content'],
                        "stream": False,
                        "keep_alive":"30m",
                        "options": {
                            #"seed": seed,
                            "num_batch":1,
                            "num_thread": 8,
                            "num_ctx":8192,
                            "num_predict": 8192,
                            # "num_gpu":0.75,
                            # "main_gpu": 0,
                        }
                    }
                  ))

    response = {
        'content': api_response.json()['response'],
        'api_obj': api_response.json()
    }

    return response
    


