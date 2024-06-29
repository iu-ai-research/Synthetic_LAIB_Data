### If you indend to run this on a server as pythong script you can run `python generate_server.py` instead.
### However please edit the relevant values in ### SELECT ###, e.g. chose model index, org_folder index and stats index.
## Import required modules
from datetime import datetime
import traceback
import json
import os
import pandas as pd
import io
import time
from api_handler import ollama_req, openai_req, anthropic_req

# Every prompt method is defined in this json dictonary according to the prompt methods design and can be found in ./2_prompt_engineering folder
PROMPT_TEMPLATE = '1prompt_templates_system_stats.json'

# Shorthand dict for prompt methods - Additional ones can be added but this should be not be touched in general and is automatically iterated over. Each of these prompt methods must be found and specified in the 
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
##############################            SELECT          #######################################
#################################################################################################
# Run generation of synthetic data prompts N times, default is 15 for research. In total this will generate 25 (expected results) * 15 iterated samples = 375 expected synthetic rows (if model is able to follow the prompt!)
N = 15

PROMPT_METHODS = list(prompt_folders.keys()) # Select key from prompt_folders dict -> e.g. 'chain_of_thought'

# When adding new providers, make sure the provider is added also and present in the api_handler.py script
# All ollama models use : for seperator, which you need to specify here, however in all further file creation it will be replaced by a -
MODEL = [
    {"name":"claude-3-opus-20240229", "provider":"anthropic"},   #0
    {"name":"claude-3-sonnet-20240229", "provider":"anthropic"}, #1
    {"name":"gemma:7b", "provider":"ollama"},                    #2
    {"name":"gpt-3.5-turbo-0125", "provider":"openai"},          #3
    {"name":"gpt-4-turbo", "provider":"openai"},                 #4
    {"name":"gpt-4o", "provider":"openai"},                      #5
    {"name":"llama2:13b", "provider":"ollama"},                  #6
    {"name":"llama3:8b", "provider":"ollama"},                   #7
    {"name":"llama3:70b", "provider":"ollama"},                  #8
    {"name":"mistral:7b", "provider":"ollama"},                  #9
    {"name":"mixtral:8x22b", "provider":"ollama"},               #10
    {"name":"phi3:medium-128k", "provider":"ollama"},            #11
    {"name":"phi3:mini-128k", "provider":"ollama"},              #12
][8] # Select model using index

# These statistical text summaries were generated from the relevant dataset using R and turned into markdown format for prompt insertion.
# Similar to the ORG_FOLDER only the d_clean_remove_small_samples_stats was used. Further research can see if including even more detailed description of each target_material will improve the overall characteristics of the synthetic data.
# The cost and time will main constraint to not try out all these different combinations. However one should not refrain from looking into these files and experimenting with their own variations of it.
STATS = [
    'd_clean_remove_small_samples_stats',                       #0
    'd_clean_remove_small_samples_stats-target_material',       #1
    'd_clean_stats',                                            #2
    'd_clean_stats-target_material',                            #3
][0] # Select a specific statistics fill which will be inserted into the prompt template

# Source folder of original nshot example rows that were sampled from the original data. For the research only the d_clean_remove_small_samples_ipr was used which was deemed to be most relevant.
ORG_FOLDER = [
    'd_clean_remove_small_samples_ipr', #0
    'd_clean_remove_small_samples_iqr', #1
    'd_full_clean_ipr',                 #2
    'd_full_clean_ipr',                 #3
][0] # Select the org folder you want to use with the index

# The sampled filenames .csv of the selected source folder - These will be the files located that were specified as the ORG_FOLDER.
ORG_SAMPLE_FILES = [
    'rs_size_5',    #0
    'rs_size_10',   #1
    'rs_size_25',   #2
    'rs_size_50',   #3
    'rs_size_100',  #4
    'rs_size_150',  #5
] # The script will automatically iterate over these sample sizes (add extra if you need in the dedicated folder)

#################################################################################################
################################       END SELECT    ############################################
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
# This function is used to transform objects that are part of the different API responses (ollama, claude, openai) into a dict that is added as metadata to the model outputs for further inspection if needed.
def convert_to_dict(obj):
    if isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {k: convert_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [convert_to_dict(i) for i in obj]
    else:
        return obj

# A fix for saving filenames using model names of ollama models. Ubuntu 24 did not have a problem, but other earlier OS version might not allow : in the filenames.
model_name_fix = MODEL['name'].replace(':','-')

# In the following loop we are iterating over the different sample files (5,10,25,50,100,150) that correspond to the N-Shot examples that are included with the prompt.
# Then in the next inner loop we iterate over every prompt method and use this as template where all relevant fields are inserted into e.g.: [INSERT EXAMPLE HERE] refering to the N-Shot Examples and [INSERT STATS HERE] refering to the descriptive statistics.
for ORG_SAMPLE_FILE in ORG_SAMPLE_FILES:
    for PROMPT_METHOD in PROMPT_METHODS:
        # Define the directory path where the generate samples will be stored. 
        # This will be ./3_llm_generation/outputs/gpt-4o/cot for example.
        directory_path = f"./../3_llm_generation/outputs/{model_name_fix}/{prompt_folders[PROMPT_METHOD]}"

        # Create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)

        # The files that already exist for the model, sample_sizes and prompt_method will be subtracted from the N count specified above (e.g. 15 required responses).
        # For example if we rerun the script, we will not unnecessarily regenerated files beyond 15 responses. This will ensure that the total amount of responses for each model is equal.
        existing_files = count_existing_files(directory_path,ORG_SAMPLE_FILE)
        samples_to_generate = max(0, N - existing_files)

        # Already got N samples, can skip generation!
        if samples_to_generate == 0:
            print(f"Already have {N} files for {PROMPT_METHOD} with sample size {ORG_SAMPLE_FILE}, skipping...")
            continue

        # If we don't have enough generated samples for the model, prompt_method & sample_size combination then generate more!
        # Print out how many are left to generate.
        print(f"Need to generate {samples_to_generate} more files for {PROMPT_METHOD} with sample size {ORG_SAMPLE_FILE}.")

        # Load the system and user prompt from the current prompt_method as variables
        system_prompt = prompts_json[PROMPT_METHOD]['system']
        user_prompt = prompts_json[PROMPT_METHOD]['user']

        # Define the prompts in a structured JSON format as how the current api_handler expects it. In the api_handler the information required by the providers api endpoint is transformed into their required format.
        # These are still the raw prompt text, where the [INSERT EXAMPLE HERE] & [INSERT STATS HERE] have to be replaced.
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

        # Read the CSV data for the N-Shot Examples, e.g. ./1_sample_preperation/org_samples/d_clean_remove_small_samples_ipr/rs_size_100.csv"
        with open(f"./../1_sample_preperation/org_samples/{ORG_FOLDER}/{ORG_SAMPLE_FILE}.csv", "r") as f:
            csv_data = f.read()

        # Read the CSV data for the statistical description, e.g. ./2_prompt_engineering/d_clean_remove_small_samples_stats" if the stats file doesn't exist replace it with an empty string.
        if STATS != '':
            with open(f"./../2_prompt_engineering/{STATS}", "r") as f:
                stats_data = f.read()
        else:
            stats_data = ''

        # Read the N-Shot Examples as CSV data into a DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        # Ensure the intensity column is numeric to handle scientific notation
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')

        # Convert each row to the desired format, some numbers are very long and they cost high amount of tokens. To reduce the token cost the row of intensity if reduced to 3 signficant digits.
        # Example:
        # target_material,target_thickness,pulse_width,energy,spot_size,intensity,power,cutoff_energy
        # aluminium,3.0,180,10.9,9.7,56830000000000000000,60560000000000,16.0
        # gold,4.0,2280,46.737,2.87,219700000000000000000,20500000000000,17.6
        # The value of 219700000000000000000 is converted to 2.197E20
        formatted_data = df.apply(
            lambda row: f"['{row['target_material']}', {row['target_thickness']}, {row['pulse_width']}, {row['energy']}, {row['spot_size']}, {row['intensity']:.3E}, {row['power']}, {row['cutoff_energy']}]", 
            axis=1
        ).tolist()

        # Combine the formatted list of rows into a single string
        example_data = "\n".join(formatted_data)

        # Replace the N-SHOT placeholder, [INSERT EXAMPLE HERE], with actual examples data in user prompt
        prompts[1]['content'] = prompts[1]['content'].replace("[INSERT EXAMPLE HERE]", example_data)

        # Replace the statistics placeholder, [INSERT STATS HERE], with actual statistics data in user prompt
        prompts[1]['content'] = prompts[1]['content'].replace("[INSERT STATS HERE]", stats_data)

        # Display the final structured prompts
        print('\n[PROMPT INPUTS]:\n',prompts)

        # Print the summary stats
        print('\n[SUMMARY STATS]:\n',stats_data)

        # Print the result
        print('\n[N-SHOT SAMPLES]:\n',example_data,'\n\n')

        # In case we get an error, retry and keep track of how many successfull API responses that were already received.
        successful_attempts = 0
        retry_count = 0
        while successful_attempts != samples_to_generate:
            try:
                ######################################################
                ####### Actual API Requests send and received ########
                ######################################################
                print(f"Starting generation {successful_attempts+1}/{samples_to_generate} with {PROMPT_METHOD} for model {model_name_fix} using {ORG_FOLDER}/{ORG_SAMPLE_FILE}.csv ...")
                
                # Get current time for appending to file_name's
                TIMESTAMP = datetime.now().isoformat()[:-7].replace(':','-')

                ##############################################################################################
                ####### Currently supported API Providers                                          ###########
                #######   - OLLAMA    (Locally downlaoded models, ollama server must be running!)  ###########
                #######   - OPENAI    (GPT-4o, GPT-3.5, etc.)                                      ###########
                #######   - ANTHROPIC (Claude, etc.)                                               ###########
                ##############################################################################################

                # Here Ollama will use the model name without fix  -> llama3:70b instead of llama3-70b (due to file systems not allowing : on some OS's)
                if(MODEL['provider'] == 'ollama'):
                    # Make API request to ollama local server - Make sure the models are downloaded and you can prompt them as usual.
                    completion =  ollama_req(model=MODEL['name'],messages=prompts)
                elif(MODEL['provider'] == 'openai'):
                    # Make API request to OpenAI API - Specifiy API KEY in .env!
                    completion = openai_req(model=MODEL['name'],messages=prompts)
                elif(MODEL['provider'] == 'anthropic'):
                    # Make API request to Anthropic API - Specifiy API KEY in .env!
                    completion = anthropic_req(model=MODEL['name'],max_tokens=4096,messages=prompts)

                ### For debugging if you want to see the whole API response
                # print(completion)

                #### The content is the text response of the large language model of interest.
                print('[RESPONSE]:\n', completion['content'])
                
                ### Turn the entire completion into a dictionary that we can add into the metadata.
                ### The content is seperated for easy access, for more info on how these outputs are structured in final form see the /outputs folder
                ### Generation metadata is saved to keep track of how the synthethic data was generated using which files.
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

                # Define the file path for saving the generated sample JSON file. e.g. ./outputs/gpt-4o/cov/gpt-4o+1prompt+d_clean_remove_small_samples_ipr+rs_size_5+cot+2024-06-10T20-07-00.json
                # The use of '+' makes it easier to split the filename by relevant information, e.g. model_name is index 0 after splitting, and sample_size (rs_size_5) is index -2 (counting backwards in list).
                file_path = f"{directory_path}/{model_name_fix}+{PROMPT_TEMPLATE.split('_')[0]}+{ORG_FOLDER}+{ORG_SAMPLE_FILE}+{prompt_folders[PROMPT_METHOD]}+{TIMESTAMP}.json"

                # Write out the file
                with open(file_path, "w") as f:
                    f.write(json.dumps(dict_completion, indent=4))

                ## Increment the amount of successfull generations for combination; model, prompt_method and sample_size.
                successful_attempts += 1

            ### Error handling, in case we fail to generate. Print out the traceback for debug and increase the retry count with sleep of 2 seconds.
            except Exception as e:
                print(f"Error occurred: {e}")
                traceback.print_exc()
                retry_count += 1
                print(f"Retrying... ({retry_count})")
                time.sleep(2)  # Optional: add a delay before retrying
        
        ### After the whole model generation is complete the final message is printed out.
        ### Adviced is to run the current cell once more, it will quickly check if all the samples are successfull or if any is missing from the target N = 15.
        print(f"Finished generating files for {PROMPT_METHOD} with sample size {ORG_SAMPLE_FILE}.")
        print(f"************* Finished {successful_attempts}/{samples_to_generate} generations! *************")