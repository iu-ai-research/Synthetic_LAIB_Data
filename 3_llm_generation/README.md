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

1. Define the same required settings as in 3-6 for the jupyter notebook.
2. Run python generate_server.py and the generation process will be printed out to the console output.
3. Once the run is complete you should see  "************* Finished #N/#N generations! *************"
4. You can run the script once more time, the script will check if all N samples were successfully created and if not generate up to the N amount.