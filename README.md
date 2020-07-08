# Virtual Data Lab (VDL)

Test drive generative models for sequential data w.r.t. to accuracy and privacy given a range of real-world and artificial datasets.

## `virtualdatalab`
* Python tooling library with following capabilities
    * data manipulation
        * `target_data_manipulation.load_convert_col_types`
            * convert a Pandas DataFrame into Common Data Format
           
    * data generation
        * `target_data_generate.generate_simple_seq_dummy`
            * sequential dummy generation
    * synthesizers
        * IdentitySynthesizer 
            * (= datacopy) 
        * FlatAutoEncoder
            * (Encoder - Decoder Fully Connected NN in PyTorch)]
        
    * metrics
        * `metrics.compare`
            * given a target dataset and synthetic dataset, compute an accuracy and privacy performance indicator 
    
    * utils for repeated experiments
        * `benchmark.benchmark` 
            * run `metrics.compare` with many synthesizers across many datasets
    

## Install 
It is recommended to install`virtualdatalab` in a Conda virtual environment if not using Google Colab.

```bash
# clone vdl
# will create conda env named vdl
conda create --name vdl
conda activate vdl
conda install pip
cd virtualdatalab/virtualdatalab
pip install -r requirements.txt
pip install . 
```

## Common Data Format 
Virtual Data Lab functionality is based on a common data format. 

The format specifies the data must be  
* Pandas DataFrame
* `id` column
* Numeric and Categorical column types 

Data is assumed to be sorted within each sequence. (as defined by the `id`)

To prepare data into common data format, please use `target_data_manipulation.load_convert_col_types()`

## Writing your own synthesizer class

All synthesizers must extend `synthesizes/base.py`. Additionally, `train` and `generate` must invoke 
parent method via `super()`. Parent functions ensure that **common data format** is respected and that models can not be 
expected to generate if they have not been trained yet. 

All synthesizer classes MUST accept the **common data format**. As a result, synthesizers are responsible for ensuring input 
is ready to be fed into a given algorithm. 

`base.generate` calls `check_is_fitted`. This simple check looks for attributes with _ naming convention. All synthesizers must
declare training attributes with this style. 



For example

```python

class MyGenerator(BaseSynthesizer):

    def train(self,data):
        super().train(data)
        data_model = some_transformation(data)
        self.train_data_ = data
        #  model is now trained
        self.data_model_ = data_model

    def generate(self,number_of_subjects):
        super().generate(self)
        generated_data = some_generation(number_of_subjects)
        
        return generated_data
```

## datasets
Preprocessed sequential datasets. Datasets are modified from original source. Sample of columns are chosen and users contain only fixed sequence length. 

Example Use:
````python
from virtualdatalab.datasets.loader import load_cdnow
cd = load_cdnow()
# pd read in csv 
````


[CD NOW](http://www.brucehardie.com/datasets/) - Transaction data of an online commerce site - `datasets/cdnow_len5.csv`
* Datetime removed 
* Fixed sequence length = 5

[1999 Czech Financial Dataset - Real Anonymized Transactions - trans.csv](https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions) Real transactions released for PKDD,99 Discovery Challenge - `datasets/berka_len50.csv`
* Fixed sequence length = 50


## Metric Definitions
`benchmark` takes a simple combination of the metrics below to output one indicator per type. 

### Accuracy

For frequency related metrics, Numeric values are sorted into 10 equally spaced quantiles. 

#### Univariate 

Max difference between target and synthetic frequencies in one column

#### Bivariate

Max difference between target and synthetic frequencies in two columns.

#### Autocorrelation

Chi-square correlation is calculatd for target and synthetic. Metric equals the max difference between each correlation
matrix.

### Privacy

#### Distance to Closest Records
The distance of each synthetic data point to its closest target data point.

We aim to have the distribution not skewed to 0, as this would indicate there is no distance between synthetic 
and target.

#### Nearest Neighbour Distance Ratio
The ratio between the closest and second closest distance of synthetic data points when 
measured against the target data set. 

An NNDR of 0 means that a given synthetic data point is only close to one point in the target, i.e an outlier. 
Thus the point would fail for privacy.     
    
## useful_notebooks  
Collection of notebooks with examples.

* [identity_synthesizer_dummy.ipynb](useful_notebooks/identity_synthesizer_dummy.ipynb)
    * Load in dummy data, synthesize with IdentitySynthesizer(samples from input data), calculate all metrics avaliable in VDL
* [google_colab_setup.ipynb](useful_notebooks/google_colab_setup.ipynb)
    * Ready made template to set up VDL on Google Colab
*  [benchmark_example.ipynb](useful_notebooks/benchmark_example.ipynb`)
    * Benchmark default settings: CDNOW + Berka, IdentitySynthesizer + FlatAutoEncoder

## Google Colab Usage
Optional prerequisites:  
Google account - if interested in saving to drive. This is recommended, since datasets can be saved and loaded from Google Drive. 

Every new notebook launched will need to reinstall VDL each time. Add the following code snippet to your Google Colab notebooks. 

```python
"""
If running on Google Colab
"""

# Mount Google Drive 

from google.colab import drive
# gdrive is google drive contents
drive.mount("/content/gdrive")
%cd gdrive/My\ Drive

import os

if os.path.isdir("vdl/virtualdatalab/virtualdatalab"):
  # repo already existing
  %cd vdl/virtualdatalab
  ! git pull 
  %cd virtualdatalab
  !pip install -r requirements.txt
  !pip install .
else:
  %mkdir vdl
  %cd vdl
  ! git clone https://github.com/mostly-ai/virtualdatalab.git
  %cd virtualdatalab/virtualdatalab
  !pip install -r requirements.txt
  !pip install .
```
Blank Template
* [google_colab_setup.ipynb](useful_notebooks/google_colab_setup.ipynb)
    * Ready made template to set up VDL on Google Colab


References:  
[Using Google Colab with Github](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC)

Notes:
Saving to Github is not possible if loading the notebook from Mostly AI public repo. 
