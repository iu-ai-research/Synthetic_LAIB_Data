# LLM Generation 

### Info

#### Run Jupyter Notebook

1. To start clean, deleted all the folders in the `./3_llm_generation/outputs` folder.
2. Make sure to define API Keys in the .env file if you want to generate using OpenAI or Anthropic API.
3. The `generate.ipynb` is a notebook where you can define the model to generate for, it has to be specified int he MODEL dictonanry and then selected using the index.
4. Specify how many samples you deem sufficient, in this research N=15 was used for all models. 25 responses are expected according to the prompt so expected rows is 15 * 25 for every model, prompt_method and sample_size combination.
5. Specify the STATS file to be used, only the `d_clean_remove_small_samples_stats` file was used in the current research.
6. Specify the ORG_FOLDER, only the `d_clean_remove_small_samples_ipr` file was used in the current research.
7. Run both cells and see the model generate responses using the defined prompt templates, n-shot samples, statistics, etc.
8. Once the run is complete you should see  "************* Finished #N/#N generations! *************"
9. You can run both cells once more time, the script will check if all N samples were successfully created and if not generate up to the N amount.


#### In case there's no access to a GUI (e.g running on a remote server) one can use the `generate_server.py` file

1. Define the same required settings as in step 3. to 6. for the jupyter notebook.
2. Run python generate_server.py and the generation process will be printed out to the console output.
3. Once the run is complete you should see  "************* Finished #N/#N generations! *************"
4. You can run the script once more time, the script will check if all N samples were successfully created and if not generate up to the N amount.

### Changing API interaction
In the `api_handler.py` additional arguments can be passed specifically for the api provider. For example one can set the num_batch to a lower number if local machine struggles or different num_thread to use more or less cpu.
`num_ctx` and `num_predict` were kept at 8192 for all models since the largest prompt token size (rs_size_150) can be nearly 8K tokens and would otherwise cutt off. Smaller models that are not trained on such large context will indeed struggle
to properly produce results beyond the context that they are able to produce or trained on. However this is accounted for in the results and also shows the different capabilities of the various models when they stop working.

### Ollama API
For more information on the values that can be passed to the ollama api in the `api_handler.py` see the official ollama git repository: 
https://github.com/ollama/ollama/blob/main/docs/api.md

### Anthropic API
Anthropic python module documentation: 
https://pypi.org/project/anthropic/

### OpenAI API
The Official OpenAI python module documentation: 
https://github.com/openai/openai-python

