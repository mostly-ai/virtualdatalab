# `virtualdatalab` (VDL)

License: [GPLv3](https://github.com/mostly-ai/virtualdatalab/blob/master/LICENSE)

VDL  to test drive generative models for sequential data w.r.t. to accuracy and privacy given a range of real-world and artificial datasets Aside from measurement capabilities, synthesizers capable of sequential data generation are also included. 


## Install 
It is recommended to install`virtualdatalab` in a Conda virtual environment to avoid interfering with other software programs. 

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


## Data Guidelines

### Input Data

Synthesizers and public functions accept only data formatted according to the following guideline. 


#### Common Data Format
* Pandas DataFrame
* `id` (denoting a subject) `sequence_pos` (order of sequences belonging to one subject) as index
* columns are either of type `numeric` or `categorical`
* single table (subject and sequence information is in single table)

`target_data_manipulation.prepare_common_data_format` is a helper function to convert a given Pandas DataFrame or CSV into the  **common data format**.

### Output Data 

Synthesizers output data in the same format as input data. 

## Useful Features

* **Data Processing**
    * `target_data_manipulation.prepare_common_data_format`
        * loads in a data source and prepares it to fit common data format
        * currently accepts `str` filepath to CSV or a Pandas DataFrame
        * data is assumed to be ordered within subject 
       
* **Datasets & Mock Data Generation**
    * `target_data_generate.generate_simple_seq_dummy`
        * sequential dummy generation
    * Preprocessed real-word datasets  
        * Datasets are formatted according to common data format when loading via helper functions 
        * Datasets are trimmed such that all users have a given fixed sequence length without padding
            * [CD NOW](http://www.brucehardie.com/datasets/) - Transaction data of an online commerce site - `datasets/data/cdnow_len5.csv`
                * Datetime converted to day of week (category) 
                * Fixed sequence length = 5
            * [1999 Czech Financial Dataset - Real Anonymized Transactions](https://data.world/lpetrocelli/czech-financial-dataset-real-anonymized-transactions) Real transactions released for PKDD,99 Discovery Challenge - `datasets/data/berka_len10.csv`
                * Fixed sequence length = 10
    
* **Included Synthesizers**
    * IdentitySynthesizer 
        * Returns a sample of data randomly selected 
    * FlatAutoEncoder
        * Encoder - Decoder Fully Connected NN in PyTorch
    
* **Metrics**
    * `metrics.compare`
        * given a target dataset and synthetic dataset, compute accuracy and privacy themed metrics (See [Metric Definitions](#metric-definitions))

* **Experimental Set-up**
    * `benchmark.benchmark` 
        * compute `metrics.compare` with many synthesizers across many datasets
        * if no datasets are passed then the default datasets are used (CDNOW + Berka)
    

## Writing your own synthesizer class

All synthesizers must extend `synthesizes/base.py`. Additionally, `train` and `generate` must invoke 
parent method via `super()`. Parent functions ensure that **common data format** is respected and that models can not be 
expected to generate if they have not been trained yet. 

All synthesizer classes MUST accept the **common data format**. As a result, synthesizers are responsible for transformation of input data. 

`base.generate` calls `check_is_fitted`. This check looks for attributes with _ naming convention. All synthesizers must
declare training attributes with this style. 

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

## Metric Definitions
`benchmark` takes a simple combination of the metrics below to output one indicator per type.

![Screenshot](readme_imgs/benchmark_example.png)

### Accuracy

* ### Frequency - Related Metrics
    * Numeric variables are binned according to 10 quantiles. Categories are binned into 20 categories if cardinality exceeds 20. 

    * #### Univariate Total Variation Distance
        * Max difference between target and synthetic frequencies in one column

    * #### Bivariate Total Variation Distance
        * Max difference between target and synthetic frequencies in two columns.

    * #### Correlation Difference

        * Chi-square correlation is calculatd for target and synthetic. Metric equals the max difference between each correlation matrix.

### Privacy

* #### Distance to Closest Records
    * The distance of each synthetic data point to its closest target data point. We aim to have the distribution not skewed to 0, as this would indicate there is no distance between synthetic and target.

* #### Nearest Neighbour Distance Ratio
    * The ratio between the closest and second closest distance of synthetic data points when 
measured against the target data set. An NNDR of 0 means that a given synthetic data point is only close to one point in the target, i.e an outlier. 
Thus the synthetic point is leaking information from the target data set. 
    
## Quick Start / Examples 
Collection of notebooks with examples.

* [identity_synthesizer_dummy.ipynb](useful_notebooks/identity_synthesizer_dummy.ipynb`) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostly-ai/virtualdatalab/blob/master/useful_notebooks/identity_synthesizer_dummy.ipynb) 
    * Load in dummy data, synthesize with IdentitySynthesizer(samples from input data), calculate all metrics avaliable in VDL  
*  [flatautoencoder_cdnow.ipynb](useful_notebooks/flatautoencoder_cdnow.ipynb`) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostly-ai/virtualdatalab/blob/master/useful_notebooks/flatautoencoder_cdnow.ipynb)
    * FlatAutoEncoder demo with CDNOW
*  [benchmark_example.ipynb](useful_notebooks/benchmark_example.ipynb`) 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostly-ai/virtualdatalab/blob/master/useful_notebooks/benchmark_example.ipynb)
    * Benchmark default settings: CDNOW + Berka, IdentitySynthesizer + FlatAutoEncoder

## Google Colab Usage
Optional prerequisites:  
Google account - if interested in saving to drive. This is recommended, since datasets can be saved and loaded from Google Drive. 
Saving to Github is not possible if loading the notebook from Mostly AI public repo. 

Every new notebook launched will need to reinstall VDL each time. Add the following code snippet to your Google Colab notebooks. 

```python
"""
If running on Google Colab
"""

%mkdir vdl
%cd vdl
! git clone https://github.com/mostly-ai/virtualdatalab.git
%cd virtualdatalab/virtualdatalab
!pip install -r requirements.txt
!pip install .
```

References:  
[Using Google Colab with Github](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC)

