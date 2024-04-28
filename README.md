# 6360Group13
A project implementation of NSG, for CS6360.001, Spring 2024

## Setup
It recommended to setup a [Python virtual environment](https://docs.python.org/3/library/venv.html) before beginning.

Run the following requirement installing command (or equivalent for your python package manager):
```shell
pip install -r requirements.txt
```

Tests were run on files in `.fvecs` format. We tested our code with the commonly used [SIFT1M](http://corpus-texmex.irisa.fr/). 

## Running

### Building a NSG

### Running a search test
In the root directory of the repository,

```shell
python src/search_test_graph.py data_file query_file nsg_path search_LK keepConstant
```

`data_file` - the path to the base vectors of the dataset
`query_file` - the path to the query vectors of the dataset
`nsg_path` - the path to the built NSG
`search_LK` - the value (L or K) to keep constant
`keepConstant` - a flag determining whether to keep L or K constant. 0 for L, otherwise K

This can generally take about a minute to run. If you want to run single search tests, instead run:

```shell
python src/nsg_serach.py data_file query_file nsg_path search_L search_K
```

`data_file` - the path to the base vectors of the dataset
`query_file` - the path to the query vectors of the dataset
`nsg_path` - the path to the built NSG
`search_L` - the L value of the search. Higher values are more accurate but cause longer runtimes.Cannot be smaller than `search_K`
`search_K` - the K nearest neighbors to find