# Evaluating Temporal Reasoning in LLMs Through Sentence Ordering

By: Chandreyi Chakraborty, Alina Polyudova, and Tejas Gururaj

Disclaimer: To prompt the model you must have an OPENAI_API_KEY as well as GROQ_API_KEY in your .env. We also recommend running on a cluster such as PACE ICE as it may run for several hours. We also ran the probe embeddings through PACE ICE as it takes up to 2 hours.

Data:
data/processed/test_processesed.csv - the data we used to prompt the models
data/finals_outputs/ - the final CSV files holding each model's inference as well as the original ordering

Scripts:
roc_functions.py - processes stories and shuffle sentences
integers_to_sentences.py - converts model's output (a 5 digit string containing numbers 1-5) into sentences
accuracy_calculations.py - implements kendall's tau and pmr
story_pos_acc.py - implement sentence index accuracy
probe_scripts/extract.py - embeds each sentence individually with Qwen3
probe_scripts/train.py - trains a logistic regression model on embeddings and labels.
probe_scripts/evaluate.py - generates the accuracy and confusion matrix of sentence label matching

Results:
model_results/ - prompted accuracies
probe_results/ - probe accuracies

Notebooks:
accuracy_model.ipynb - where we ran the accuracy scripts to get the numerical outputs
accuracy_graphs.ipynb - where we generated the graphs that are stored in results from the numerical outputs
model_test.ipynb - where we first tested prompting the models

```
conda create -n storyorder python=3.10

conda activate storyorder

pip install -r requirements.txt

python -m ipykernel install --user --name storyorder --display-name "storyorder"
```