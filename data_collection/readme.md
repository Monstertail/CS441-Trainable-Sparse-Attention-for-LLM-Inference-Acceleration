## Data pipeline

### Step 1: Collect a small CS 441 Q&A seed set

Download the official review questions and practice questions from CS 441.  
Clean and convert them into two JSON files:

- `cs441_quiz_original.json`
- `cs441_practice_original.json`

Each file should contain a list of question–answer pairs.

### Step 2: Design the prompt

We use an LLM to expand this small seed set into a larger synthetic dataset.  
The following prompt is passed to the LLM together with a few examples from the original JSON files.

```
Prompt="""
Generate 200 single-choice questions related to the following CS 441 course topics. Use the knowledge points and question format found in the cs441_train_original.json file as examples for style and content.

CS 441 Course Topic List:
- Fundamentals of Learning
- K-NN Classification, Data Representation
- K-NN Regression, Generalization
- Search and Clustering
- Dimensionality reduction: PCA, embeddings
- Linear regression, regularization
- Linear classifiers: logistic regression, SVM
- Naïve Bayes Classifier
- EM and Latent Variables
- Density estimation: MoG, Hists, KDE
- Outliers and Robust Estimation
- Decision Trees
- Ensembles and Random Forests
- Stochastic Gradient Descent
- MLPs and Backprop
- CNNs and Keys to Deep Learning
- Deep Learning Optimization and Computer Vision
- Words and Attention
- Transformers in Language and Vision
- Foundation Models: CLIP and GPT
- Ethics and Impact of AI
- Bias in AI, Fair ML
- Audio and 1D Signals
- Reinforcement Learning

Each question must have options labeled a, b, c, d, etc., and the answer provided in the "answer" field must be the single correct option letter. Ensure the generated questions cover a diverse range of topics and strictly adhere to the following JSON format:

[
  {"question": "Question text with options labeled a, b, c...", "answer": "a"},
  {"question": "...", "answer": "b"},
  ...
]

"""
```
### Step 3: Generate synthetic data with an LLM

Feed the above prompt and a subset of examples from  
`cs441_quiz_original.json` and `cs441_practice_original.json` into an LLM  
(Gemini in my case). The model outputs a larger JSON file:

- `cs441_synthetic.json`

This file is a list of objects of the form:

```json
{"question": "…", "answer": "a"}

```

### Step 4: Split the synthetic dataset into train and test sets

Use `create_synthetic_data_with_llm.py` to randomly split  
`cs441_synthetic.json` into training and test subsets. You can choose your own train/test ratio.

Example:

```bash
python create_synthetic_data_with_llm.py \
  --input path/to/cs441_synthetic.json \
  --train-ratio 0.8 \
  --seed 42 \
  --output-dir path/to/output_dir
```

- `--train-ratio` controls the fraction of data used for training (default: `0.8`).
- `--seed` sets the random seed for reproducible splits.
- `--output-dir` is where the script writes:
  - `cs441_synthetic_train.json`
  - `cs441_synthetic_test.json`