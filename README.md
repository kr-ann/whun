# whun: Humor generation project
## GPT-2 Joke Generation
This is something that works.

**Preprocessing**
* Data preprocessing.ipynb is the notebook with data visualization and preprocessing;
* preprocess_before_train.py is the script for the final data cleaning before the training.

**Model training**
* run_clm.py is the script for generative models training;
* runs.txt contains info on how the training was run for all the models;
* generate.py is the script for joke generation given a model.

Humor generation report.pdf contains description of experiments and a brief analysis of results.

## Sequence to Sequence Joke Generation using PEGASUS Transformer
This is something that does not work. 

**Model Training**
* train_category_jokes.py trains based on joke category. Input is given as the category of the joke.
* train_title_jokes.py trains based on joke title. Input is the title of the joke. 

**Generation**
* generate_category_joke.py can be used to generate jokes based on joke category. Enter whichever joke category you can think as input.
* generate_title_joke.py can be used to generate jokes based on titles given a model. Just enter whatever title you can think as input.

**Python Notebooks**
* PEGASUS_category_jokes.ipynb contains the whole workflow from pre-processing, to training and generating the jokes based on joke categories.
* PEGASUS_title_jokes.ipynb contains the whole workflow from pre-processing, to training and generating the jokes based on joke title.
