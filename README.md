# 6360Group13
A project implementation of [NSG](https://vldb.org/pvldb/vol12/p461-fu.pdf), for CS6360.001, Spring 2024.

## Setup
It recommended to setup a [Python virtual environment](https://docs.python.org/3/library/venv.html) before beginning.

Run the following requirement installing command (or equivalent for your python package manager):
```shell
pip install -r requirements.txt
```

Tests were run on files in `.fvecs` format. We tested our code with the commonly used [SIFT10K and SIFT1M](http://corpus-texmex.irisa.fr/). These are `tar` files transferred through FTP, so some form of `wget` and `tar` are necessary to download and extract these files.

If using `wget`, the following command will install the SIFT10K files.
```shell
wget -c ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
```
Extraction with tar:
```shell
tar -xzvf siftsmall.tar.gz
```

Files for SIFT10K are already provided in `./siftsmall`

## Running

### Building a NSG

Building a NSG requires a kNN. Our project used [efanna_graph](https://github.com/ZJULearning/efanna_graph) to create ours. However, there is one kNN provided for SIFT10K in `benchmarks/sift.50NN.graph`.

```shell
python src/driver.py data_file nn_graph_path L R C save_graph_file
```

`data_file` - the path to the dataset file
`nn_graph_path` - the path to the kNN graph file
`L` - the size of the candidate pool (larger L can improve accuracy but uses more memory and computation)
`R` - the number of nearest expanded nodes to consider during the graph building process
`C` - the number of neighbors in the final graph for each node
save_graph_file - the path where the built NSG should be saved

### Running a search test
In the root directory of the repository,

```shell
python src/search_test_graph.py base_file query_file nsg_file search_constant setConstantK
```

`base_file` - the path to the base vectors of the dataset  
`query_file` - the path to the query vectors of the dataset  
`nsg_file` - the path to the built NSG  
`search_constant` - the value (L or K) to keep constant  
`setConstantK` - a flag determining whether to keep L or K constant. 0 for L, otherwise K

This can generally take about a minute to run. If you want to run single search tests, instead run:

```shell
python src/nsg_search.py base_file query_file nsg_file L K
```

`base_file` - the path to the base vectors of the dataset  
`query_file` - the path to the query vectors of the dataset  
`nsg_file` - the path to the built NSG  
`L` - the L value (candidate neighbor pool size) of the search. Higher values are more accurate but cause longer runtimes. Cannot be smaller than `K`  
`K` - the K nearest neighbors to find  

Note that prebuilt NSGs for SIFT10K can be found in `./siftsmall`.

## Experiments

For searching, we ran the following command to generate the graphs.

For constant L:
```shell
python src/search_test_graph.py siftsmall/siftsmall_base.fvecs siftsmall/siftsmall_query.fvecs siftsmall/siftsmall0.nsg 50 0
```
For constant K:
```shell
python src/search_test_graph.py siftsmall/siftsmall_base.fvecs siftsmall/siftsmall_query.fvecs siftsmall/siftsmall0.nsg 50 1
```