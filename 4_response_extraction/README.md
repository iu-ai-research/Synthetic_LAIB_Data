# Response Extraction 

### Info

#### Run Jupyter Notebooks

By running `extract_v5.ipynb` all the generated responses in `./3_llm_generation/outputs` will be processed.
Different Regex Patterns that we're most frequently found through iterative elimination of captured and uncaptured prompts were found ideal.

1. Every model defined in MODELS will be searched for and attempted to extract the relevant response data. You can add your own models here or remove the ones not used.
2. Every sample size defined will be searched for, also add the relevant ones or remove the one not used.
3. Every prompt_method in PROMPT_SHORT_DICT will be searched for, it's important that the short name match with the generated data.
4. Run both cells, this might take some time, however the process is designed to be multi-threaded, depending on the availble CPU threads the time will vary.
5. The script will extract unique regex rows that are validated to follow target_material of only letters, followed by 7 numeric values which also account for scientific notations.
6. Final extracted data will be written to a json file which can be loaded as a dataframe. If the dataframe is unable to load directly using pd.from_json first load the file into python as a dictionary and then pass it to pandas to create the dataframe.
