# whun: Humor generation project
## GPT-2 Joke Generation
This is something that works.

**Preprocessing**
* `Data preprocessing.ipynb` is the notebook with data visualization and preprocessing;
* `preprocess_before_train.py` is the script for the final data cleaning before the training.

**Model training**
* `run_clm.py` is the script for generative models training;
* `runs.txt` contains info on how the training was run for all the models;
* `generate.py` is the script for joke generation given a model.

`GPT-2 Report.pdf` contains description of experiments and a brief analysis of results.

## Sequence to Sequence Joke Generation using PEGASUS Transformer
This is something that does not work. 

**Model Training**
* `train_category_jokes.py` trains based on joke category. Input is given as the category of the joke.
* `train_title_jokes.py trains` based on joke title. Input is the title of the joke. 

**Generation**
* `generate_category_joke.py` can be used to generate jokes based on joke category. Enter whichever joke category you can think as input.
* `generate_title_joke.py` can be used to generate jokes based on titles given a model. Just enter whatever title you can think as input.

**Python Notebooks**
* `PEGASUS_category_jokes.ipynb` contains the whole workflow from pre-processing, to training and generating the jokes based on joke categories.
* `PEGASUS_title_jokes.ipynb` contains the whole workflow from pre-processing, to training and generating the jokes based on joke title.

`Seq2Seq-Joke-Generation` contains report of the experiments.

## About Dataset

We used the joke datasets from this github repo: https://github.com/taivop/joke-dataset

```
----------------------------------------------
reddit_jokes.json |  195K jokes | 7.40M tokens
stupidstuff.json  | 3.77K jokes |  396K tokens
wocka.json        | 10.0K jokes | 1.11M tokens
----------------------------------------------
TOTAL             |  208K jokes | 8.91M tokens
----------------------------------------------
```

**Some more statistics...**

* `wocka.json`contains 9k (unique) joke titles and 24 joke categories.
* `reddit_jokes.json` contains 117k unique joke titles. This dataset does not have joke categories.
* `stupidstuff.json` contains 43 joke categories. This dataset does not have joke titles.
* Both, `stupidstuff.json` and `wocka.json` contain a combine number of 59 (unique) joke categories.

## References

* Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.
* Zhang, J., Zhao, Y., Saleh, M., & Liu, P. (2020, November). Pegasus: Pre-training with extracted gap-sentences for abstractive summarization. In International Conference on Machine Learning (pp. 11328-11339). PMLR.
* Pungas, Taivo. (2017). A dataset of English plaintext jokes. GitHub Repository.
