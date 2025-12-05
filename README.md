# Evaluating Temporal Reasoning in LLMs Through Sentence Ordering

By: Chandreyi Chakraborty, Alina Polyudova, and Tejas Gururaj


Disclaimer:  
To prompt the model you must have an OPENAI_API_KEY as well as a GROQ_API_KEY in your .env.  
We also recommend running on a cluster such as PACE ICE as it may run for several hours.  
We also ran the probe embeddings through PACE ICE as it takes up to 2 hours.


Data:

data/processed/train_processesed.csv - The data we used to prompt the models  

data/finals_outputs/ - The final CSV files holding each model's inference (model_reordered) as well as the original ordering (gold) per story


Scripts:

roc_functions.py - Processes stories and shuffle sentences  

integers_to_sentences.py - Converts model's output (a 5 digit string containing numbers 1–5) into sentences  

accuracy_calculations.py - Implements Kendall's Tau and PMR  

story_pos_acc.py - Implements sentence index accuracy  

probe_scripts/extract.py - Embeds each sentence individually with Qwen3  

probe_scripts/train.py - Trains a logistic regression model on embeddings and labels  

probe_scripts/evaluate.py - Generates the accuracy and confusion matrix of sentence label matching  


Results:

model_results/ - Prompted accuracies  

probe_results/ - Probe accuracies  


Notebooks:

accuracy_model.ipynb - Where we ran the accuracy scripts to get the numerical outputs  

accuracy_graphs.ipynb - Where we generated the graphs stored in results from the numerical outputs  

model_test.ipynb - Where we first tested prompting the models  


Environment Setup:
```
conda create -n storyorder python=3.10  

conda activate storyorder  

pip install -r requirements.txt  

python -m ipykernel install --user --name storyorder --display-name "storyorder"
```
