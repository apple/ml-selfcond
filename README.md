# Self-Conditioning Pre-Trained Language Models
<p align="center">
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

This software project accompanies the paper [Self-Conditioning Pre-Trained Language Models](https://arxiv.org/abs/2110.02802). 

## Installation

The requirements are listed in [frozen_requirements.txt](frozen_requirements.txt). The code has been tested using `Python 3.8` on MacOS and Linux Ubuntu 18.04. Run the following for installation:


#### Create a virtual environment
```bash
cd <path_to_this_project>
python3 -m venv env  # make sure Python3 >= 3.6
source env/bin/activate
pip install -U pip wheel
```

#### Install selfcond (recommended for reproducibility)
```bash
pip install -r frozen_requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

#### Testing
Using the `"not slow"` marker will avoid downloading models from the transformers repository.
```bash
pytest -m "not slow"
```

To test the full pipeline:
```bash
pytest
```

-----
## 1. Finding Expert Units

We provide instructions to read responses from a model given a dataset, to compute expertise results per concept and to aggregate results.

The models are fetched from the [HuggingFace Transformers repository](https://huggingface.co/transformers/). 
We currently support `[gpt2, gpt-medium, gpt2-large, gpt2-xl]`.


### 1.1 Reading responses from a model

A small dataset with 1 concept (`football`) is provided in `assets/football` for model GPT2-Medium. It contains a file `concept_list.csv` with the concepts in the dataset,
 as well as the actual data inside a folder called `sense` (since the concepts are of the WordNet _sense_ type, see paper). 
 The data for each concept is provided as a `json` file, with `positive` and `negative` sentences.
If other concepts were to be added, of a different type, we would save them in a different folder, with the appropriate name.


Run the following script to collect responses from a model. If you have a GPU, this step will run much faster than on CPU.
Choose the model version with `--model-name-or-path`, for example `--model-name-or-path gpt-medium`.

```bash
python scripts/compute_responses.py \
--model-name-or-path gpt2-medium \
--data-path assets/football \
--responses-path /tmp/path_to_save_responses \
--device cuda
```

> The script above assumes a file `concept_list.csv` inside the dataset path. 
> If we want to run the script in specific concepts, pass argument `--concepts` with comma
> separated concepts and specifying the type. For example: `--concepts sense/football-1_04_00__,[some_other_concept],...`

The responses will be saved inside `path_to_save_responses/gpt2-medium/sense/[concept]/responses`.

### 1.2 Computing expertise per concept

The following script will compute the expertise per unit for each concept. 
The expertise is defined as the Average Precision (AP) achieved by a unit when its responses are considered 
prediction scores for the concept sentences.

```bash
python scripts/compute_expertise.py \
--root-dir /tmp/path_to_save_responses \
--model-name gpt2-medium \
--concepts assets/football/concept_list.csv
```

The expertise results are saved as a CSV file in `path_to_save_responses/gpt2-medium/sense/[concept]/expertise`.
Column `ap` contains the expertise measured for each model unit and column `on_p50` contains the median response of each unit to the positive sentences (Sec 4.2 in paper). 


## 2. Open ended self-conditioned generation

In this step, the above computed expertise is used to generate sentences conditioned on a concept. We provide a script for open ended generation of sentences: `scripts/generate_seq.py`. It has several parameters to control the decoding strategy, sequence length, expertise related, etc. See `--help` for details.

Here we give a simple example that generates sentences with concept `football` for which we obtained the expertise in steps 1.x:

```bash
python scripts/generate_seq.py \
--model-name-or-path gpt2-medium \
--expertise my_results/gpt2-medium/sense/football-1_04_00__/expertise/expertise.csv \
# Generate sentences of 20 tokens
--length 20 \
# The initial context passed to the model
--prompt "Once upon a time" \
# Generate 10 sentences with random seeds ranging from 0 -> 9
--seed 0 10 \
# Final softmax temperature
--temperature 1.0 \
# Expert units are sorted according to AP
--metric ap \
# And experts are forced using the median expected value (named on_p50)
--forcing on_p50  \
# Condition the top 50 expert units
--num-units 50 \
# Just show the results, otherwise would save as a csv
--no-save
```

## 3. Replicating the paper results

### 3.1 Generate sentences

**NOTE:** First of all, the expertise for concepts `woman` and `man` should be computed using steps 1.x and the datasets in `assets/gender_bias`.

We provide a script that distributes the generation across multiple GPUs (if available) to speed up the experimentation (`scripts/generate_batch.py`).

The following example will obtain the generated sentences used in the paper.

```bash
python scripts/generate_batch.py \
--concept some_path/gpt2-medium/sense/woman-1_18_00__/expertise/expertise.csv \
--device cuda \
--prompts occupations \
--method ours \
--folder generated
```
The generated sentences will be stored in files `generated/gen_sentences_[concept]_[context].csv` (one file per concept per context).

#### Running FUDGE

We provide a patch to run [FUDGE](https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation) and 
obtain generated sentences compatible with our repository.
To run FUDGE we must first clone the code and apply the patch `git-patches/fudge.patch`:

```bash
git clone https://github.com/yangkevin2/naacl-2021-fudge-controlled-generation.git fudge
cd fudge
git checkout fbedf820c306c5b3cbf684dc414b2464fc603222
# Apply patch
git am $SELFCOND/git-patches/fudge.patch

# Create a new virtualenv, since PPLM uses a frozen version of transformers
virtualenv env_fudge --python=python3.6
. env_fudge/bin/activate
pip install -r requirements.txt
```

Then, use script `run_batch.py` with the desired arguments. In the patch, we provide the BoW and prompts used in the paper in the directory `topic_data`.
.


#### Running PPLM-BoW

Note that `scripts/generate_batch.py` also allows running [PPLM-BoW](https://github.com/uber-research/PPLM).
To run PPLM-BoW we must first clone the code and apply the patch `git-patches/pplm.patch`.

```bash
git clone https://github.com/uber-research/PPLM.git
cd PPLM
git checkout e236b8989322128360182d29a79944627957ad47
# Apply patch
git am $SELFCOND/git-patches/pplm.patch

# Create a new virtualenv, since PPLM uses a frozen version of transformers
virtualenv env_pplm --python=python3.6
. env_pplm/bin/activate
pip install -r requirements.txt
```

Then, use the argument `--method pplm-bow` when calling `scripts/generate_batch.py`.


### 3.2 Probability of generating specific words

The following step will aggregate results and obtain the probabilities of specific words appearing after the prompt.
For example, in the example we compute `p(he,his | do(woman, k))` and store it in a file `p_he_woman.csv`.

```bash
python scripts/compute_frequency.py 
# All the files with sentences to consider.
--sentences-df "some_path/generated/gen_sentences_woman_*.csv"  
# Number of characters to consider after the prompt.
--num-chars 5  
# Method can be selfcond, fudge or pplm
--method selfcond  
--out-file results/selfcond/p_he_woman.csv 
--words "he;his"
```

In the paper, we do this step for words `he;his` and `she;her`, and for all sentences generated for `man` and `woman`, obtaining:
* `p_he_woman.csv`
* `p_she_woman.csv`
* `p_he_man.csv`
* `p_she_man.csv`

Additionally, in our paper we report the probabilities of generating words `woman;women` and `man:men` when conditioning on `woman` or `man` respecitvely. These files should be saved as:
* `p_woman_woman.csv`
* `p_man_man.csv`

> NOTE: In order to be able to run step 3.4, save the csv files in `results/[method]` as in the example.

### 3.3 Computing perplexity

We also provide a script to compute the perplexity of generated sentences after generation. As explained in the paper, for that we use a different model family, in this case `openai-gpt`.

```bash
python scripts/compute_perplexity.py 
--model-name-or-path openai-gpt 
--sentences-df some_path/generated/gen_sentences_man*.csv
--device cuda 
# Method can be selfcond, fudge or pplm
--method selfcond 
```

The results will be saved in a file with the same name as `--sentences-df` but ending with `_ppl.csv` instead.

#### Aggregating the perplexity

We have an additional script that aggregates the perplexity computed above. 
Example of usage:

```bash
python scripts/aggregate_perplexities.py 
--ppl-woman some_path/generated/gen_sentences_woman*ppl.csv
--ppl-man some_path/generated/gen_sentences_man*ppl.csv
# Method can be selfcond, fudge or pplm
--method selfcond 
--out-dir some_path/results
```

The aggregated perplexities will be saved as `pl_woman.csv` and `ppl_man.csv` in `results/[method]`.


### 3.4 Computing Self-BLEU score

Run the following script, that will compute the Self-BLEU score for all the generated sentences passed as `--sentences-df`.
To speed up the computation (at the expense of a higher variance in the score) one can reduce both `--num-sample` and `--num-reps`.

```bash
python scripts/compute_selfbleu.py
--sentences-df some_path/generated/gen_sentences_*.csv
# Method can be selfcond, fudge or pplm
--method selfcond 
--out-dir some_path/results
# Number of sentences randomly sampled to compute the score
--num-sample 100
# Number of repetitions performed, the reported score will be the average
--num-reps 1000
--ngram 3 4
```

The Self-BLEU scores will be saved in `selfbleu_woman.csv` and `selfbleu_man.csv` in `results/[method]`.

### 3.5 The figures

The script `all_plots.py` will produce all the figures in the paper. 
It assumes all the results in `csv` format in a single folder. In the steps above we used `results` as our main results folder.
All figures will be saved in the directory specified by `-o`.

Example of usage:
```bash
python scripts/all_plots.py  -i results -o figures
```

> If you need dark figures, set the variable `DARK_PLOTS=True` in the script.

----

## Citation

```bibtex
@article{suau2022selfcond,
  title={Self-Conditioning Pre-Trained Language Models},
  author={Suau, Xavier and Zappella, Luca and Apostoloff, Nicholas},
  journal={International Conference on Machine Learning},
  year={2022}
}
```
----

## Contact

> Xavier Suau Cuadros (xsuaucuadros@apple.com)

> Luca Zappella (lzappella@apple.com)

> Nick Apostoloff (napostoloff@apple.com)
