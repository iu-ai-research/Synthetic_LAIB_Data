from datetime import datetime
import traceback
import json
import os
import pandas as pd
import io
import time
from api_handler import ollama_req, openai_req, anthropic_req

# Shorthand dict
prompt_folders = {
    'chain_of_thought':'cot',                       #0
    'skeleton_of_thought':'sot',                    #1
    'self_consistency':'sc',                        #2
    'generated_knowledge':'gk',                     #3
    'least_to_most':'ltm',                          #4
    'chain_of_verification':'cov',                  #5
    'step_back_prompting':'sbp',                    #6
    'rephrase_and_respond':'rar',                   #7
    'emotion_prompt':'em',                          #8
    'directional_stimuli':'ds',                     #9
    'recursive_criticism_and_improvement':'rcai',   #10
    'reverse_prompting':'rp'                        #11
}

#################################################################################################
#################################################################################################
#################################################################################################

# Run generation of synthetic data prompts N times.
N = 15

PROMPT_METHODS = list(prompt_folders.keys()) # Select key from prompt_folders dict -> e.g. 'chain_of_thought'
PROMPT_TEMPLATE = '1prompt_templates_system_stats.json'

# When adding new providers, make sure the provider is added also and present in the api_handler.py script
# All ollama models use : for seperator, which you need to specify here, however in all further file creation it will be replaced by a -
MODELS = [
    {"name":"claude-3-opus-20240229", "provider":"anthropic"},   #0
    {"name":"claude-3-sonnet-20240229", "provider":"anthropic"}, #1
    {"name":"falcon:40b", "provider":"ollama"},                  #2
    {"name":"falcon:180b", "provider":"ollama"},                 #3
    {"name":"gemma:7b", "provider":"ollama"},                    #4
    {"name":"gpt-3.5-turbo-0125", "provider":"openai"},          #5
    {"name":"gpt-4-turbo", "provider":"openai"},                 #6
    {"name":"gpt-4o", "provider":"openai"},                      #7
    {"name":"llama2:13b", "provider":"ollama"},                  #8
    {"name":"llama3:8b", "provider":"ollama"},                   #9
    {"name":"llama3:70b", "provider":"ollama"},                  #10
    {"name":"mistral:7b", "provider":"ollama"},                  #11
    {"name":"mixtral:8x22b", "provider":"ollama"},               #12
    {"name":"phi3:medium-128k", "provider":"ollama"},            #13
    {"name":"phi3:mini-128k", "provider":"ollama"},              #14
    {"name":"qwen2:72b", "provider":"ollama"},                   #15
][4]

# Source folder of original samples
ORG_FOLDER = [
    'd_clean_remove_small_samples_ipr', #0
    'd_clean_remove_small_samples_iqr', #1
    'd_full_clean_ipr',                 #2
    'd_full_clean_ipr',                 #3
][0] # Selected num

# The sampled filenames .csv of the selected source folder
ORG_SAMPLE_FILES = [
    'rs_size_5',    #0
    'rs_size_10',   #1
    'rs_size_25',   #2
    'rs_size_50',   #3
    'rs_size_100',  #4
    'rs_size_150',  #5
    # 'rs_size_250',  #6
] # Selected num

# These statistical text summaries were generated from the relevant dataset using R and turned into markdown format for prompt insertion.
STATS = [
    'd_clean_remove_small_samples_stats',                       #0
    'd_clean_remove_small_samples_stats-target_material',       #1
    'd_clean_stats',                                            #2
    'd_clean_stats-target_material',                            #3
][0] # Selected num

#################################################################################################
#################################################################################################
#################################################################################################

# Load the prompt
with open(f"./../2_prompt_engineering/{PROMPT_TEMPLATE}","r") as f:
    prompts_json = json.loads(f.read())

# Count existing files for combination model, sample_size, prompt_method..
def count_existing_files(directory, sample_size):
    """Count files in a directory that exactly match a specified sample size using string split."""
    count = 0
    sample_size_tag = sample_size  # Create the tag to look for in the filename parts
    for name in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, name)):
            parts = name.split('+')
            if sample_size_tag in parts:  # Check if the sample size tag is in the list of parts
                count += 1
    return count

# Function to recursively convert an object to a dictionary
def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    else:
        return obj

model_name_fix = MODEL['name'].replace(':','-')
for ORG_SAMPLE_FILE in ORG_SAMPLE_FILES:
    for PROMPT_METHOD in PROMPT_METHODS:
        # Define the directory path
        directory_path = f"./../3_llm_generation/outputs/{model_name_fix}/{prompt_folders[PROMPT_METHOD]}"

        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        existing_files = count_existing_files(directory_path,ORG_SAMPLE_FILE)
        samples_to_generate = max(0, N - existing_files)

        # Already got N samples, can skip generation!
        if samples_to_generate == 0:
            print(f"Already have {N} files for {PROMPT_METHOD} with sample size {ORG_SAMPLE_FILE}, skipping...")
            continue

        # Rest of your code for data preparation and generation goes here
        print(f"Need to generate {samples_to_generate} more files for {PROMPT_METHOD} with sample size {ORG_SAMPLE_FILE}.")

        system_prompt = prompts_json[PROMPT_METHOD]['system']
        user_prompt = prompts_json[PROMPT_METHOD]['user']

        # Define the prompts in a structured JSON format
        prompts = [
            {
                "role":"system",
                "content":system_prompt
            },
            {
                "role":"user",
                "content":user_prompt
            }
        ]

        # Read the CSV data from the file
        with open(f"./../1_sample_preperation/org_samples/{ORG_FOLDER}/{ORG_SAMPLE_FILE}.csv", "r") as f:
            csv_data = f.read()

        # Read the CSV data from the file
        if STATS != '':
            with open(f"./../2_prompt_engineering/{STATS}", "r") as f:
                stats_data = f.read()
        else:
            stats_data = ''

        # Read the CSV data into a DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Ensure the intensity column is numeric to handle scientific notation
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')

        # Convert each row to the desired format
        formatted_data = df.apply(
            lambda row: f"['{row['target_material']}', {row['target_thickness']}, {row['pulse_width']}, {row['energy']}, {row['spot_size']}, {row['intensity']:.3E}, {row['power']}, {row['cutoff_energy']}]", 
            axis=1
        ).tolist()

        # Combine the formatted rows into a single string
        example_data = "\n".join(formatted_data)

        # Replace placeholder with actual example data in user prompt
        prompts[1]['content'] = prompts[1]['content'].replace("[INSERT EXAMPLE HERE]", example_data)

        # Replace placeholder with actual example data in user prompt
        prompts[1]['content'] = prompts[1]['content'].replace("[INSERT STATS HERE]", stats_data)

        # Display the final structured prompts
        print('\n[PROMPT INPUTS]:\n',prompts)

        # Print the summary stats
        print('\n[SUMMARY STATS]:\n',stats_data)

        # Print the result
        print('\n[N-SHOT SAMPLES]:\n',example_data,'\n\n')

        successful_attempts = 0
        retry_count = 0
        while successful_attempts != samples_to_generate:
            try:
                print(f"Starting generation {successful_attempts+1}/{samples_to_generate} with {PROMPT_METHOD} for model {model_name_fix} using {ORG_FOLDER}/{ORG_SAMPLE_FILE}.csv ...")
                # Get current time
                TIMESTAMP = datetime.now().isoformat()[:-7].replace(':','-')

                # Here Ollama will use the model name without fix  -> llama3:70b instead of llama3-70b
                if(MODEL['provider'] == 'ollama'):
                    # Make API request to ollama local server - Make sure the models are downloaded and you can prompt them as usual.
                    completion =  ollama_req(model=MODEL['name'],messages=prompts)
                elif(MODEL['provider'] == 'openai'):
                    # Make API request to OpenAI API - Specifiy API KEY in .env!
                    completion = openai_req(model=MODEL['name'],messages=prompts)
                elif(MODEL['provider'] == 'anthropic'):
                    # Make API request to Anthropic API - Specifiy API KEY in .env!
                    completion = anthropic_req(model=MODEL['name'],max_tokens=4096,messages=prompts)

                # print(completion)

                print('[RESPONSE]:\n', completion['content'])
                    
                dict_completion = convert_to_dict(completion)
                dict_completion['inputs'] = prompts
                dict_completion['metadata_gen'] = {
                    'N': samples_to_generate,
                    'MODEL': model_name_fix,
                    'ORG_FOLDER': ORG_FOLDER,
                    'ORG_SAMPLE_FILE': ORG_SAMPLE_FILE,
                    'PROMPT_METHOD': PROMPT_METHOD,
                    'PROMPT_TEMPLATE': PROMPT_TEMPLATE,
                    'STATS': STATS,
                }

                # Define the file path
                file_path = f"{directory_path}/{model_name_fix}+{PROMPT_TEMPLATE.split('_')[0]}+{ORG_FOLDER}+{ORG_SAMPLE_FILE}+{prompt_folders[PROMPT_METHOD]}+{TIMESTAMP}.json"

                # Write to the file
                with open(file_path, "w") as f:
                    f.write(json.dumps(dict_completion, indent=4))

                successful_attempts += 1

            except Exception as e:
                print(f"Error occurred: {e}")
                traceback.print_exc()
                retry_count += 1
                print(f"Retrying... ({retry_count})")
                time.sleep(2)  # Optional: add a delay before retrying
        
        print(f"Finished generating files for {PROMPT_METHOD} with sample size {ORG_SAMPLE_FILE}.")
        print(f"************* Finished {successful_attempts}/{samples_to_generate} generations! *************")