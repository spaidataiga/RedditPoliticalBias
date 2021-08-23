# RedditPoliticalBias

This is the entire repository of code and data used to create my Master's Thesis. The data-set is clean and ready for use, and the python files have not been edited from their use in my thesis. However, as these files have been compiled from local and remote sources, filepaths will need to be altered to run properly. Please contact the original author for guidance if needed.

## Repository Structure

The structure of the repository is explained below:

* data
	* This contains the data used to conduct this thesis. It has been processed as described in the Data section of my thesis. Large data-sets are saved in a .7z compressed version so it could be uploaded onto github. It contains the following subfolders:
	* *classifier* : contains the data splits used in the classifier as well as the pre-trained 100-dimension embeddings and word substitition list.
	* *data_subsets* : smaller subsections of the all_comments.csv
	* *external_datasets* : data sourced externally used, or considered for use in my thesis
	* *interim_results* : interim data files (e.g. for PMI)
* figures
	* This contains the various figures included in my thesis, in addition to several that were not included.
* notebooks
	* This contains jupyter notebooks (and some RStudio code) used to create the figures, and conduct some of the statistical analyses shown in this thesis. These notebooks were working documents and may be difficult to run, as they are not all organised and many cells may be out of order. Please contact the original author (me) for help if the code in these notebooks needs to be run.
* python
	* This contains folders of the codes used to process the data, train the model, and run the analyses.

## Environments

The code in this document must be run with two separate environments. If the document requires environment 1, it will say at the top of the file. Otherwise, all code can be run on environment 2. The required packages for both environments are listed below.

### *Environment1*

blis            0.2.4
boto3           1.17.33
botocore        1.20.33
certifi         2020.12.5
chardet         4.0.0
cymem           2.0.5
Cython          0.29.22
en-core-web-md  2.1.0
idna            2.10
jmespath        0.10.0
jsonschema      2.6.0
murmurhash      1.0.5
neuralcoref     4.0
numpy           1.19.5
pandas          1.1.5
pip             21.0.1
plac            0.9.6
pmaw            1.0.5
preshed         2.0.1
python-dateutil 2.8.1
pytz            2021.1
requests        2.25.1
s3transfer      0.3.6
setuptools      39.2.0
six             1.15.0
spacy           2.1.0
srsly           1.0.5
thinc           7.0.8
tqdm            4.59.0
urllib3         1.26.4
wasabi          0.8.2

### *Environment2*

backcall           0.2.0
blis               0.7.4
bpemb              0.3.2
catalogue          2.0.4
certifi            2020.12.5
chardet            4.0.0
click              7.1.2
cloudpickle        1.6.0
colorama           0.4.4
contextvars        2.4
cycler             0.10.0
cymem              2.0.5
dataclasses        0.8
decorator          4.4.2
Deprecated         1.2.12
en-core-web-lg     3.0.0
filelock           3.0.12
flair              0.8.0.post1
ftfy               5.9
future             0.18.2
gdown              3.12.2
gensim             3.8.3
huggingface-hub    0.0.8
hyperopt           0.2.5
idna               2.10
immutables         0.15
importlib-metadata 3.10.0
ipython            7.16.1
ipython-genutils   0.2.0
Janome             0.4.1
jedi               0.18.0
Jinja2             3.0.1
joblib             1.0.1
kiwisolver         1.3.1
konoha             4.6.4
langdetect         1.0.8
lxml               4.6.3
MarkupSafe         2.0.1
matplotlib         3.3.4
mpld3              0.3
murmurhash         1.0.5
networkx           2.5.1
nltk               3.6.1
numpy              1.19.5
overrides          3.1.0
packaging          20.9
pandas             1.1.5
parso              0.8.2
pathy              0.5.2
pexpect            4.8.0
pickleshare        0.7.5
Pillow             8.2.0
pip                21.1.2
preshed            3.0.5
prompt-toolkit     3.0.19
ptyprocess         0.7.0
pydantic           1.7.4
Pygments           2.9.0
pyparsing          2.4.7
PySocks            1.7.1
python-dateutil    2.8.1
pytz               2021.1
regex              2021.4.4
REL                0.0.1
requests           2.25.1
sacremoses         0.0.44
scikit-learn       0.24.1
scipy              1.5.4
segtok             1.5.10
sentencepiece      0.1.95
setuptools         57.0.0
six                1.15.0
smart-open         3.0.0
spacy              3.0.6
spacy-alignments   0.8.3
spacy-legacy       3.0.5
spacy-transformers 1.0.2
sqlitedict         1.7.0
srsly              2.4.1
tabulate           0.8.9
thinc              8.0.3
threadpoolctl      2.1.0
tokenizers         0.10.2
torch              1.9.0+cu102
torchaudio         0.9.0
torchtext          0.10.0
torchvision        0.10.0+cu102
tqdm               4.60.0
traitlets          4.3.3
transformers       4.5.0
typer              0.3.2
typing-extensions  3.7.4.3
Unidecode          1.2.0
urllib3            1.26.4
wasabi             0.8.2
wcwidth            0.2.5
wheel              0.36.2
wikimapper         0.1.5
wrapt              1.12.1
zipp               3.4.1
