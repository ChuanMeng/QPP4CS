# Query Performance Prediction for Conversational Search (QPP4CS) 
![](https://api.visitorbadge.io/api/VisitorHit?user=ChuanMeng&repo=QPP4CS&countColor=%237B1E7A)

This is the repository for the papers:
- [Query Performance Prediction: From Ad-hoc to Conversational Search](https://dl.acm.org/doi/abs/10.1145/3539618.3591919) (SIGIR 2023)
- [Performance Prediction for Conversational Search Using Perplexities of Query Rewrites](https://ceur-ws.org/Vol-3366/paper-05.pdf) (QPP++ 2023)

The repository offers the implementation of a comprehensive collection of pre- and post-retrieval query performance prediction (QPP) methods, all integrated within a unified Python/Pytorch framework. It would be an ideal package for anyone interested in conducting research into QPP for ad-hoc or conversational search.

We kindly ask you to cite our papers if you find this repository useful: 
```
@inproceedings{meng2023query,
 author = {Meng, Chuan and Arabzadeh, Negar and Aliannejadi, Mohammad and de Rijke, Maarten},
 title = {Query Performance Prediction: From Ad-hoc to Conversational Search},
 booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
 pages = {2583–2593},
 year = {2023},
 url = {https://doi.org/10.1145/3539618.3591919},
 doi = {10.1145/3539618.3591919},
}

@inproceedings{meng2023Performance,
 author = {Meng, Chuan and Aliannejadi, Mohammad and de Rijke, Maarten},
 title = {Performance Prediction for Conversational Search Using Perplexities of Query Rewrites},
 booktitle = {Proceedings of the The QPP++ 2023: Query Performance Prediction and Its Evaluation in New Tasks Workshop co-located with The 45th European Conference on Information Retrieval},
 year = {2023},
 pages = {25--28}
}
```

This repository allows the replication of all results reported in the papers. In particular, it is organized as follows:
- [Prerequisites](#Prerequisites)
- [Data Preparation](#Data-Preparation)
  - [Raw File Download](#Raw-File-Download) 
  - [Preprocessing](#Preprocessing)
  - [Indexing](#Indexing)
  - [Generating Query Rewrites](#Generating-Query-Rewrites)
  - [Preparing Run Files](#Preparing-Run-Files)
  - [Preparing Actual Performance Files](#Preparing-Actual-Performance-Files)
- [Replicating Results](#Replicating-Results)
  - [Pre-retrieval QPP methods](#Pre-retrieval-QPP-methods)
  - [Perplexity-based pre-retrieval QPP framework](#Perplexity-based-pre-retrieval-QPP-framework)
  - [Evaluation for pre-retrieval QPP methods](#Evaluation-for-pre-retrieval-QPP-methods)
  - [Post-retrieval unsupervised QPP methods](#Post-retrieval-unsupervised-QPP-methods)
  - [Post-retrieval supervised QPP methods](#Post-retrieval-supervised-QPP-methods)
  - [Evaluation for post-retrieval QPP methods](#Evaluation-for-post-retrieval-QPP-methods)
- [Plots](#Plots)

Note that for ease of use, we already uploaded the predicted performance files for all QPP methods reported in our paper. See [here](./results_in_papers). 

## Prerequisites
We recommend running all the things in a Linux environment. 
Please create a conda environment with all required packages, and activate the environment by the following commands:
```
$ conda env create -f environment.yaml
$ conda activate QPP4CS
```

## Data Preparation
Query performance prediction for conversational search needs **query rewrites**, **retrieval run files** and **actual performance files**.
To this end, we need to [download raw dataset files](#Raw-File-Download), [conduct preprocessing](#Preprocessing), [build indexes](#Indexing), [perform query rewriting](#Generating-Query-Rewrites), [perform retrieval](#Preparing-Run-Files), and [generate actual performance files](#Preparing-Actual-Performance-Files).

> For ease of use, you can directly download the `dataset` folder [here](https://drive.google.com/file/d/1Oz2w6HgaX1e_92WueyqB3whWLsGcwyjc/view?usp=sharing), which contains the preprocessed qrels files, query rewrites, retrieval run files and actual performance files for CAsT-19 & CAsT-20 and OR-QuAC datasets; please put the unzipped `dataset` folder in the current folder. Raw files for CAsT-19 & CAsT-20 and OR-QuAC are not included and they need to be downloaded by the following procedure. The collections and indexes for CAsT-19 & CAsT-20 and OR-QuAC are too large and so they also need to be produced by the following procedure.

### Raw File Download

#### CAsT-19 & CAsT-20 
CAsT-19 and CAsT-20 share the same collection.
Use the following commands to download the collection files (the MS MARCO Passage Ranking collection, the TREC CAR paragraph collection v2.0 and the MARCO duplicate file) of CAsT-19 & CAsT-20:
```bash
mkdir datasets/
mkdir datasets/cast-19-20/ 
mkdir datasets/cast-19-20/raw    
wget -P datasets/cast-19-20/raw https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget -P datasets/cast-19-20/raw http://trec-car.cs.unh.edu/datareleases/v2.0/paragraphCorpus.v2.0.tar.xz
wget -P datasets/cast-19-20/raw http://boston.lti.cs.cmu.edu/Services/treccast19/duplicate_list_v1.0.txt
tar zxvf datasets/cast-19-20/raw/collection.tar.gz -C datasets/cast-19-20/raw/
tar xvJf datasets/cast-19-20/raw/paragraphCorpus.v2.0.tar.xz -C datasets/cast-19-20/raw/
mv datasets/cast-19-20/raw/collection.tsv datasets/cast-19-20/raw/msmarco.tsv
```
These files are stored in `./datasets/cast-19-20/raw`.

Use the following commands to download query and qrels files of CAsT-19 & CAsT-20:
```bash
wget -P datasets/cast-19-20/raw https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json
wget -P datasets/cast-19-20/raw https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv
wget -P datasets/cast-19-20/raw https://trec.nist.gov/data/cast/2019qrels.txt --no-check-certificate
wget -P datasets/cast-19-20/raw https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_automatic_evaluation_topics_v1.0.json
wget -P datasets/cast-19-20/raw https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_manual_evaluation_topics_v1.0.json
wget -P datasets/cast-19-20/raw https://trec.nist.gov/data/cast/2020qrels.txt --no-check-certificate
```
These files are stored in `./datasets/cast-19-20/raw`.

#### OR-QuAC
Use the following commands to download the collection, query and qrels files of OR-QuAC:
```bash
mkdir datasets/or-quac/
mkdir datasets/or-quac/raw
wget -P datasets/or-quac/raw https://ciir.cs.umass.edu/downloads/ORConvQA/all_blocks.txt.gz --no-check-certificate
wget -P datasets/or-quac/raw https://ciir.cs.umass.edu/downloads/ORConvQA/qrels.txt.gz --no-check-certificate
gzip -d datasets/or-quac/raw/*.txt.gz
wget -P datasets/or-quac/raw https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/train.txt --no-check-certificate
wget -P datasets/or-quac/raw https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/dev.txt --no-check-certificate
wget -P datasets/or-quac/raw https://ciir.cs.umass.edu/downloads/ORConvQA/preprocessed/test.txt --no-check-certificate
```
These files are stored in `./datasets/or-quac/raw`.

### Preprocessing
#### CAsT-19 & CAsT-20 
Note that our preprocessing follows [Yu et al.](https://github.com/thunlp/ConvDR/tree/main/data).
Use the following command to preprocess the collection, query and qrels files of CAsT-19 and CAsT-20. 
```bash
python preprocess.py --dataset cast-19-20
```
The preprocessed collection file is stored in `datasets/cast-19-20/jsonl`, and the qrels files of CAsT-19 (**cast-19.qrels.txt**) and CAsT-20 (**cast-20.qrels.txt**) are stored in `datasets/cast-19-20/qrels`.
This preprocessing process also produces human-rewritten query files of CAsT-19 (**cast-19.queries-manual.tsv**) and CAsT-20 (**cast-20.queries-manual.tsv**), which are stored in `datasets/cast-19-20/queries`.

#### OR-QuAC
Use the following command to preprocess the collection, query and qrels files of OR-QuAC:
```bash
python preprocess.py --dataset or-quac
```
The preprocessed collection file is stored in `./datasets/or-quac/jsonl`, and the qrels file (**or-quac.qrels.txt**) is stored in `./datasets/or-quac/qrels`.
This preprocessing process also produces human-rewritten query files of the training set (**or-quac-train.queries-manual.tsv**), development set (**or-quac-dev.queries-manual.tsv**) and test set (**or-quac-test.queries-manual.tsv**), which are stored in `./datasets/or-quac/queries`.

### Indexing
We use [Pyserini](https://github.com/castorini/pyserini) to conduct indexing and retrieval. 
We follow the default Pyserini setting to index collections.
Use the following commands to index the collection of CAsT-19 & CAsT-20:
```bash
python -m pyserini.index.lucene --collection JsonCollection --generator DefaultLuceneDocumentGenerator --threads 16 -input datasets/cast-19-20/jsonl -index datasets/cast-19-20/index --storePositions --storeDocvectors --storeRaw 
```
The index is stored in `./datasets/cast-19-20/index`.

Use the following commands to index the collection of OR-QuAC:
```bash
python -m pyserini.index.lucene --collection JsonCollection --generator DefaultLuceneDocumentGenerator --threads 16 -input datasets/or-quac/jsonl -index datasets/or-quac/index  --storePositions --storeDocvectors --storeRaw
```
The index is stored in `./datasets/or-quac/index`.

### Generating Query Rewrites
We consider three kinds of query rewrites: **T5-based query rewrites**, **QuReTeC-based query rewrites** and **human-rewritten queries**.
- We already obtain the human-rewritten queries for CAsT-19, CAsT-20 and OR-QuAC during [Preprocessing](#Preprocessing). 
- We use [the T5 rewriter](https://huggingface.co/castorini/t5-base-canard) released by [Lin et al.](https://dl.acm.org/doi/abs/10.1145/3446426) to generate T5-based query rewrites on CAsT-19, CAsT-20 and OR-QuAC.
- We use [QuReTeC](https://github.com/nickvosk/sigir2020-query-resolution) proposed by [Voskarides et al.](https://dl.acm.org/doi/10.1145/3397271.3401130) to generate QuReTeC-based query rewrites on OR-QuAC, but directly use the QuReTeC-based query rewrites on CAsT-19 and CAsT-20 generated by [Vakulenko et al.](https://github.com/svakulenk0/cast_evaluation).

Note that we recommend using **GPU** to execute the following commands.
#### CAsT-19 
Use the following command to generate the T5-based query rewrites for CAsT-19:
```bash
python -u  query_rewriters.py  --dataset cast-19 --rewriter T5  --response_num 0
```
The produced T5-based query rewrites for CAsT-19 (**cast-19.queries-T5-Q.tsv**) are stored in `./datasets/cast-19-20/queries`.

Use the following command to directly fetch the QuReTeC-based query rewrites on CAsT-19 from [Vakulenko et al.](https://github.com/svakulenk0/cast_evaluation) and unify the file format:
```bash
wget -P datasets/cast-19-20/queries https://raw.githubusercontent.com/svakulenk0/cast_evaluation/main/rewrites/2019/5_QuReTeC_Q.tsv
python preprocess_QuReTeC.py --input_path datasets/cast-19-20/queries/5_QuReTeC_Q.tsv --output_path datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv
```
The produced QuReTeC-based query rewrites for CAsT-19 (**cast-19.queries-QuReTeC-Q.tsv**) are stored in `./datasets/cast-19-20/queries`.

#### CAsT-20
On CAsT-20, we follow [Vakulenko et al.](https://github.com/svakulenk0/cast_evaluation) and prepend the **automatic canonical response** to the query of the previous conversational turn when generating T5-based query-rewrites:
```bash
python -u  query_rewriters.py  --dataset cast-20 --rewriter T5  --response_num 1
```
The produced T5-based query rewrites for CAsT-20 (**cast-20.queries-T5-QA.tsv**) are stored in `./datasets/cast-19-20/queries`.

Similar to CAsT-19, we directly fetch the QuReTeC-based query rewrites on CAsT-20 from [Vakulenko et al.](https://github.com/svakulenk0/cast_evaluation):
```bash
wget -P datasets/cast-19-20/queries https://raw.githubusercontent.com/svakulenk0/cast_evaluation/main/rewrites/2020/5_QuReTeC_QnA.tsv
python preprocess_QuReTeC.py --input_path datasets/cast-19-20/queries/5_QuReTeC_QnA.tsv --output_path datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv
```
The produced QuReTeC-based query rewrites for CAsT-20 (**cast-20.queries-QuReTeC-QA.tsv**) are stored in `./datasets/cast-19-20/queries`.

#### OR-QuAC
On OR-QuAC, following [Yu et al.](https://github.com/thunlp/ConvDR), we do not feed the ground-truth answers in conversational history into query rewriters. 
[Qu at al.](https://dl.acm.org/doi/abs/10.1145/3397271.3401110) also argue that it is unrealistic for a model to access the ground-truth answers.
Thus we only feed the current query and queries in the conversational history into the T5 and QuReTeC-based query rewriters.
Note that please first download [the QuReTeC checkpoint](https://drive.google.com/file/d/1BKvRoKnbjWWne8Cp-dfLkQ_6s3SIte1f/view) released by [Voskarides et al.](https://github.com/nickvosk/sigir2020-query-resolution) and then unzip the model checkpoint in the current folder.

Use the following commands to generate the T5 and QuReTeC-based query rewrites on the development set of OR-QuAC:
```bash
python -u  query_rewriters.py  --dataset or-quac-dev  --rewriter T5 --response_num=0 
python -u  query_rewriters.py  --dataset or-quac-dev  --rewriter QuReTeC --response_num=0  --model_path models/191790_50
```
Note that `models/191790_50` is the path to [the QuReTeC checkpoint](https://drive.google.com/file/d/1BKvRoKnbjWWne8Cp-dfLkQ_6s3SIte1f/view).
The produced T5 (**or-quac-dev.queries-T5-Q.tsv**) and QuReTeC-based (**or-quac-dev.queries-QuReTeC-Q.tsv**) query rewrites on the development set of OR-QuAC are stored in `./datasets/or-quac/queries`.

Similarly, use the following commands to generate the T5 and QuReTeC-based query rewrites on the test set of OR-QuAC:
```bash
python -u  query_rewriters.py  --dataset or-quac-test  --rewriter T5 --response_num=0 
python -u  query_rewriters.py  --dataset or-quac-test  --rewriter QuReTeC --response_num=0  --model_path models/191790_50
```
The produced T5 (**or-quac-test.queries-T5-Q.tsv**) and QuReTeC-based (**or-quac-test.queries-QuReTeC-Q.tsv**) query rewrites on the test set of OR-QuAC are stored in `./datasets/or-quac/queries`.

Note that [the T5 rewriter](https://huggingface.co/castorini/t5-base-canard) and [QuReTeC](https://github.com/nickvosk/sigir2020-query-resolution) are trained on the training set of [CANARD](https://aclanthology.org/D19-1605/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter).
However, the queries in the training set of OR-QuAC and in the training set of [CANARD](https://aclanthology.org/D19-1605/?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter) are the same.
Thus, it is unreasonable to run [the T5 rewriter](https://huggingface.co/castorini/t5-base-canard) and [QuReTeC](https://github.com/nickvosk/sigir2020-query-resolution) on the training set of OR-QuAC.

### Preparing Run Files
We consider two groups of conversational search methods, namely: BM25 with different query rewriters and a conversational dense retrieval method.
- For the former, we feed [Pyserini](https://github.com/castorini/pyserini) BM25 with the default setting (k1=0.9, b=0.4) with the T5-based query rewrites, QuReTeC-based query rewrites and human-rewritten queries.
- For the latter, we consider [ConvDR](https://dl.acm.org/doi/abs/10.1145/3404835.3462856) and rerun ConvDR on CAsT-19, CAsT-20 as well as the OR-QuAC using the [code](https://github.com/thunlp/ConvDR) released by the author.

Note that in order to get the run files of ConvDR, please download the `dataset` folder [here](https://drive.google.com/file/d/1Oz2w6HgaX1e_92WueyqB3whWLsGcwyjc/view?usp=sharing) and put the unzipped `dataset` folder in the current folder.
We already put run files of ConvDR on CAsT-19 (**cast-19.run-ConvDR-1000.txt**), CAsT-20 (**cast-20.run-ConvDR-1000.txt**) in `./datasets/cast-19-20/runs`, while we put the run files of ConvDR on the development set (**or-quac-dev.run-ConvDR-1000.txt**) and the test set (**or-quac-test.run-ConvDR-1000.txt**) of the OR-QuAC in `./datasets/or-quac/runs`.

The following shows how to perform retrieval using BM25 given different query rewrites.
#### CAsT-19
Use the following commands to perform retrieval using BM25 given the T5-based query rewrites, QuReTeC-based query rewrites and human-rewritten queries on CAsT-19:
```bash
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-19.run-T5-Q-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-19.run-QuReTeC-Q-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-19.queries-manual.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-19.run-manual-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
```
The run files of BM25 with T5-based query rewrites (**cast-19.run-T5-Q-bm25-1000.txt**), QuReTeC-based query rewrites (**cast-19.run-QuReTeC-Q-bm25-1000.txt**) and human-rewritten queries (**cast-19.run-manual-bm25-1000.txt**) on CAsT-19 are stored in `./datasets/cast-19-20/runs`.

#### CAsT-20
Similarly, use the following commands to perform retrieval using BM25 given the three kinds of query rewrites on CAsT-20:
```bash
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-20.run-T5-QA-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-20.run-QuReTeC-QA-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-20.queries-manual.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-20.run-manual-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64 
```
The run files of BM25 with T5-based query rewrites (**cast-20.run-T5-QA-bm25-1000.txt**), QuReTeC-based query rewrites (**cast-20.run-QuReTeC-QA-bm25-1000.txt**) and human-rewritten queries (**cast-20.run-manual-bm25-1000.txt**) on CAsT-20 are stored in `./datasets/cast-19-20/runs`.

#### OR-QuAC
Use the following commands to perform retrieval using BM25 given the three kinds of query rewrites on the development set of OR-QuAC:
```bash
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-dev.run-T5-Q-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-dev.queries-QuReTeC-Q.tsv  --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-dev.run-QuReTeC-Q-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64 
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-dev.queries-manual.tsv --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-dev.run-manual-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
```
The run files of BM25 with T5-based query rewrites (**or-quac-dev.run-T5-Q-bm25-1000.txt**), QuReTeC-based query rewrites (**or-quac-dev.run-QuReTeC-Q-bm25-1000.txt**) and human-rewritten queries (**or-quac-dev.run-manual-bm25-1000.txt**) on the development set of OR-QuAC are stored in `./datasets/or-quac/runs`.

Similarly, use the following commands to perform retrieval using BM25 given the three kinds of query rewrites on the test set of OR-QuAC:
```bash
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv  --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-test.run-T5-Q-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv  --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-test.run-QuReTeC-Q-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64 
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-test.queries-manual.tsv  --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-test.run-manual-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
```
The run files of BM25 with T5-based query rewrites (**or-quac-test.run-T5-Q-bm25-1000.txt**), QuReTeC-based query rewrites (**or-quac-test.run-QuReTeC-Q-bm25-1000.txt**) and human-rewritten queries (**or-quac-test.run-manual-bm25-1000.txt**) on the test set of OR-QuAC are stored in `./datasets/or-quac/runs`.

We perform retrieval using BM25 given the human-rewritten queries on the training set of OR-QuAC:
```bash
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-train.queries-manual.tsv --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-train.run-manual-bm25-1000.txt --bm25 --hits 1000 --threads 16 --batch-size 64
```
The run file of BM25 with the human-rewritten queries (**or-quac-train.run-manual-bm25-1000.txt**) on the training set of OR-QuAC is stored in `./datasets/or-quac/runs`.

### Preparing Actual Performance Files
For each run file, we generate an actual performance file that contains scores of various IR metrics (e.g., nDCG@3) for each query.
Note that following [Dalton et al.](https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.C.pdf), we use relevance scale ≥ 2 as positive for all binary relevance metrics (e.g., Recall) on CAsT-20.

#### CAsT-19
Use the following commands to generate actual performance files for all the run files on CAsT-19:
```bash
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-19.run-T5-Q-bm25-1000.txt --qrel datasets/cast-19-20/qrels/cast-19.qrels.txt
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-19.run-QuReTeC-Q-bm25-1000.txt --qrel datasets/cast-19-20/qrels/cast-19.qrels.txt
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-19.run-manual-bm25-1000.txt --qrel datasets/cast-19-20/qrels/cast-19.qrels.txt
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt --qrel datasets/cast-19-20/qrels/cast-19.qrels.txt
```
We get the following actual performance files in `./datasets/cast-19-20/actual_performance` through the above commands:
**cast-19.actual-performance-run-T5-Q-bm25-1000.json**, **cast-19.actual-performance-run-QuReTeC-Q-bm25-1000.json**, **cast-19.actual-performance-run-manual-bm25-1000.json**, and **cast-19.actual-performance-run-ConvDR-1000.json**.

#### CAsT-20
Use the following commands to generate actual performance files for all the run files on CAsT-20:
```bash
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-20.run-T5-QA-bm25-1000.txt --qrel datasets/cast-19-20/qrels/cast-20.qrels.txt
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-20.run-QuReTeC-QA-bm25-1000.txt --qrel datasets/cast-19-20/qrels/cast-20.qrels.txt
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-20.run-manual-bm25-1000.txt --qrel datasets/cast-19-20/qrels/cast-20.qrels.txt
python -u evaluation_retrieval.py --run datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt --qrel datasets/cast-19-20/qrels/cast-20.qrels.txt
```
We get the following actual performance files in `./datasets/cast-19-20/actual_performance` through the above commands:
**cast-20.actual-performance-run-T5-QA-bm25-1000.json**, **cast-20.actual-performance-run-QuReTeC-QA-bm25-1000.json**, **cast-20.actual-performance-run-manual-bm25-1000.json**, and **cast-20.actual-performance-run-ConvDR-1000.json**.

#### OR-QuAC
Use the following commands to generate actual performance files for all the run files on the development set of OR-QuAC:
```bash
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-dev.run-T5-Q-bm25-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-dev.run-QuReTeC-Q-bm25-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-dev.run-manual-bm25-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-dev.run-ConvDR-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
```
We get the following actual performance files in `datasets/or-quac/actual_performance` through the above commands:
**or-quac-dev.actual-performance-run-T5-Q-bm25-1000.json**, **or-quac-dev.actual-performance-run-QuReTeC-Q-bm25-1000.json**, **or-quac-dev.actual-performance-run-manual-bm25-1000.json**, and **or-quac-dev.actual-performance-run-ConvDR-1000.json**.

Use the following commands to generate actual performance files for all the run files on the test set of OR-QuAC:
```bash
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-test.run-T5-Q-bm25-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-test.run-QuReTeC-Q-bm25-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-test.run-manual-bm25-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
```
We get the following actual performance files in `datasets/or-quac/actual_performance` through the above commands:
**or-quac-test.actual-performance-run-T5-Q-bm25-1000.json**, **or-quac-test.actual-performance-run-QuReTeC-Q-bm25-1000.json**, **or-quac-test.actual-performance-run-manual-bm25-1000.json**, and **or-quac-test.actual-performance-run-ConvDR-1000.json**.

Use the following command to generate actual performance files for the run file **or-quac-train.run-manual-bm25-1000.txt** on the training set of OR-QuAC:
```bash
python -u evaluation_retrieval.py --run datasets/or-quac/runs/or-quac-train.run-manual-bm25-1000.txt --qrel datasets/or-quac/qrels/or-quac.qrels.txt
```
Through the above command, we get the actual performance file **or-quac-train.actual-performance-run-manual-bm25-1000.json** in `./datasets/or-quac/actual_performance`.

## Replicating Results
> For ease of use, we already uploaded the predicted performance files for all QPP methods reported in our paper. See [here](./results_in_papers). 

### Pre-retrieval QPP methods
#### Precomputation
Some of the pre-retrieval QPP methods (VAR and PMI) would take a very long time to run. 
In order to reduce the time consumption, we first conduct precomputation on the two collections. 
The files of precomputation would be saved in the path `./output/pre-retrieval/`. 
CAsT-19 and CAsT-20 share the same collection. Run the following command to do precomputation on the shared collection.
```bash
python -u ./unsupervisedQPP/pre_retrieval.py \
--mode precomputation \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--query_path_2 ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--output_path ./output/pre-retrieval
```
Run the following command to do precomputation on the collection of OR-QUAC.
```bash
python -u ./unsupervisedQPP/pre_retrieval.py \
--mode precomputation \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--query_path_2 ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--output_path ./output/pre-retrieval
```
#### Computation
Run the following commands to run pre-retrieval QPP methods (QS, SCS, avgICTF, IDF, PMI, SCQ, VAR) on the CAsT-19, CAsT-20 and the test set of the OR-QUAC datasets:
```bash
python -u ./unsupervisedQPP/pre_retrieval.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/pre-retrieval

python -u ./unsupervisedQPP/pre_retrieval.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/pre-retrieval

python -u ./unsupervisedQPP/pre_retrieval.py \
--mode baselines \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/pre-retrieval
```
The output files of these methods would be saved in the path `./output/pre-retrieval/`. 
The output file would include ```qid \t predicted performance``` per line.

### Perplexity-based pre-retrieval QPP framework
#### Perplexity Computation
Run the following commands to compute the perplexities of query rewrites on the CAsT-19, CAsT-20 and the test set of the OR-QUAC datasets:
```bash
python -u ./unsupervisedQPP/pre_retrieval.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/pre-retrieval \
--LM gpt2-xl

python -u ./unsupervisedQPP/pre_retrieval.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/pre-retrieval \
--LM gpt2-xl

python -u ./unsupervisedQPP/pre_retrieval.py \
--mode ppl \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/pre-retrieval \
--LM gpt2-xl
```
The output files would be saved in the path `./output/pre-retrieval/`. 

#### Perplexity-based Pre-retrieval QPP framework
Run the following commands to run the Perplexity-based pre-retrieval QPP framework (PPL-QPP) on the CAsT-19, CAsT-20 and the test set of the OR-QUAC datasets:
```bash
python -u ./unsupervisedQPP/pre_retrieval.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/pre-retrieval \
--LM gpt2-xl \
--qpp_names VAR-std-sum \
--alpha 0.1

python -u ./unsupervisedQPP/pre_retrieval.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/pre-retrieval \
--LM gpt2-xl \
--qpp_names SCQ-avg \
--alpha 0.2

python -u ./unsupervisedQPP/pre_retrieval.py \
--mode PPL-QPP \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/pre-retrieval \
--LM gpt2-xl \
--qpp_names VAR-std-sum \
--alpha 0
```
The output files of PPL-QPP would be saved in the path `./output/pre-retrieval/`.

### Evaluation for pre-retrieval QPP methods
Run the following commands to evaluate all pre-retrieval QPP methods and PPL-QPP for estimating the retrieval quality of T5-based query rewrites+BM25 in terms of Pearson, Kendall, and Spearman correlation coefficients:
```bash
python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/cast-19.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3 # can be set to "ndcg@3", "ndcg@100" or "recall@100"

python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/cast-20.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/or-quac-test.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3
```
```target_metric``` here can be set to "ndcg@3", "ndcg@100" or "recall@100". The files showing the evaluation results would be saved in the path `./output/pre-retrieval/`. 

### Post-retrieval unsupervised QPP methods
The following is used to run the post-retrieval unsupervised QPP methods: Clarity, WIG, NQC, SMV, σ<sub>max</sub> and n(σ<sub>x%</sub>).

#### Assessing BM25 on CAsT-19
When assessing BM25, QPP methods and BM25 always share the same query rewrites. 
Use the following commands to estimate the retrieval quality of T5-based, QuReTeC-based and human-written query rewrites+BM25 on CAsT-19:
```bash
 python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--run_path ./datasets/cast-19-20/runs/cast-19.run-T5-Q-bm25-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/post-retrieval/ 

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--run_path ./datasets/cast-19-20/runs/cast-19.run-QuReTeC-Q-bm25-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/post-retrieval/ 

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--run_path ./datasets/cast-19-20/runs/cast-19.run-manual-bm25-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/post-retrieval/
```
The output files of these methods would be saved in the path `./output/post-retrieval/`. 
The output file would include ```qid \t predicted performance``` per line.

#### Assessing BM25 on CAsT-20
Similarly, use the following commands to estimate the retrieval quality of BM25 with T5-based, QuReTeC-based and human-written query rewrites on CAsT-20:
```bash
python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--run_path ./datasets/cast-19-20/runs/cast-20.run-T5-QA-bm25-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/post-retrieval/ 

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--run_path ./datasets/cast-19-20/runs/cast-20.run-QuReTeC-QA-bm25-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/post-retrieval/

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--run_path ./datasets/cast-19-20/runs/cast-20.run-manual-bm25-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/post-retrieval/
```
#### Assessing BM25 on OR-QuAC
Similarly, use the following commands to estimate the retrieval quality of BM25 with T5-based, QuReTeC-based and human-written query rewrites on the test set of OR-QuAC:
```bash
python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--run_path ./datasets/or-quac/runs/or-quac-test.run-T5-Q-bm25-1000.txt \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/post-retrieval/

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--run_path ./datasets/or-quac/runs/or-quac-test.run-QuReTeC-Q-bm25-1000.txt \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/post-retrieval/

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--run_path ./datasets/or-quac/runs/or-quac-test.run-manual-bm25-1000.txt \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/post-retrieval/ 
```
#### Assessing ConvDR on CAsT-19
ConvDR has a specially-trained query encoder to encode raw utterances. 
QPP methods designed for ad-hoc search do not have a special module to understand raw utterances.
When estimating the retrieval quality of ConvDR, we consider three types of inputs to QPP methods to help QPP methods understand the current query, namely T5-based, QuReTeC-based and human-written query rewrites. 
Use the following commands to estimate the retrieval quality of ConvDR on CAsT-19:
```bash
python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/post-retrieval/

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/post-retrieval/

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--output_path ./output/post-retrieval/ 
```
The output files of these methods would be saved in the path `./output/post-retrieval/`.

#### Assessing ConvDR on CAsT-20
Similarly, use the following commands to estimate the retrieval quality of ConvDR on CAsT-20:
```bash
python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/post-retrieval/

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/post-retrieval/

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--output_path ./output/post-retrieval/ 
```

#### Assessing ConvDR on OR-QuAC
Similarly, use the following commands to estimate the retrieval quality of BM25 with T5-based, QuReTeC-based and human-written query rewrites on the test set of OR-QuAC:
```bash
python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/post-retrieval/ 

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/post-retrieval/ 

python -u unsupervisedQPP/post_retrieval.py \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--output_path ./output/post-retrieval/ 
```
### Post-retrieval supervised QPP methods
We consider three state-of-the-art supervised QPP methods, namely [NQAQPP](https://dl.acm.org/doi/abs/10.1145/3341981.3344249), [BERTQPP](https://dl.acm.org/doi/abs/10.1145/3459637.3482063) and [qppBERTPL](https://dl.acm.org/doi/abs/10.1145/3477495.3531821).
Note that we recommend using GPU to execute the following commands. 
We use an **NVIDIA RTX A6000 GPU** to train all supervised methods with the same random seed 42. 
Please use the same device and random seed if you would like to precisely replicate the results reported in our paper.

Note that, during training, qppBERTPL is a classification-based model and does not learn to approximate scores of a specific IR metric. However, regression-based models (e.g., NQAQPP, BERTQPP) will learn to estimate the retrieval quality in terms of a specific IR metric, such as nDCG@3, nDCG@100 or Recall@100.

Note that for all experiments on OR-QuAC, we first train a QPP model on the training set of OR-QuAC, and then conduct inference on the test set of OR-QuAC.
During training, we always train a QPP model to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC.
There are two reasons.
First, when training a QPP method to learn to estimate the retrieval quality of T5-based or QuReTeC-based query rewrites+BM25, we need to train the QPP method on the run files of T5-based or QuReTeC-based query rewrites+BM25 on the training set of OR-QuAC. 
However, [the T5 rewriter](https://huggingface.co/castorini/t5-base-canard) and [QuReTeC](https://github.com/nickvosk/sigir2020-query-resolution) we use in this paper were trained over the queries in the training set of OR-QuAC. 
Thus it is unreasonable to run them on the training set of OR-QuAC. 
Second, when training a QPP method to learn to predict the retrieval quality of ConvDR, we still need to train the QPP method on the run file of ConvDR on the training set of OR-QuAC. 
Because ConvDR was trained on the training set of OR-QuAC, there would be a great shift between the run files of ConvDR on the training set and the test set of OR-QuAC. 
Thus, it is also unreasonable to train a QPP method using the run file of ConvDR on the training set of OR-QuAC.
So we always train a QPP method to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC.

Note that all experiments on CAsT-19 and CAsT-20 are conducted using 5-fold cross-validation.

#### NQAQPP on OR-QuAC  
Use the following command to train NQAQPP to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC in terms of nDCG@3:
```bash
python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-train.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-train.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-train.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \ # can be set to "ndcg@3", "ndcg@100" or "recall@100"
--epoch_num 1  \
--interval 1000
```
The training process would produce checkpoints, which are stored in `./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/`. 
`--target_metric` can be set to "ndcg@3", "ndcg@100" or "recall@100"; This variable would not impact qppBERTPL during training.

After training, use the following commands to run NQAQPP to estimate the retrieval quality of T5-based, QuReTeC-based and human-written query rewrites+BM25 on the test set of OR-QuAC:
```bash
python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-T5-Q-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-QuReTeC-Q-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference
```
The output files of NQAQPP would be saved in the path `./output/post-retrieval/`. 
The output file would include ```qid \t predicted performance``` per line.
`--checkpoint_name` shows the saved checkpoints to be inferred.
Please make sure the value of `--target_metric` is consistent with the one used during training.
For example, if `--checkpoint_name` is `or-quac-train.manual-bm25-1000.manual-NQAQPP-recall@100`, which means that the model is trained to approximate the values of Recall@100 during training, `--target_metric` used during inference should be set to `recall@100`.

ConvDR has a specially-trained query encoder to encode raw utterances. 
QPP methods designed for ad-hoc search do not have a special module to understand raw utterances.
When estimating the retrieval quality of ConvDR, we consider three types of inputs to a QPP method to help the QPP method understand the current query, namely T5-based, QuReTeC-based and human-written query rewrites.
Use the following commands to run NQAQPP to estimate the retrieval quality of ConvDR on the test set of OR-QuAC:
```bash
python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference
```

#### BERTQPP on OR-QuAC
Likewise, use the following command to train BERTQPP to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC in terms of nDCG@3:
```bash
python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-train.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-train.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-train.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000
```
The training process would produce checkpoints, which are stored in `./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/`.

After training, use the following commands to run BERTQPP to estimate the retrieval quality of T5-based, QuReTeC-based and human-written query rewrites+BM25 on the test set of OR-QuAC:
```bash
python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-T5-Q-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-QuReTeC-Q-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference
```
The output files of BERTQPP would be saved in the path `./output/post-retrieval/`.

Use the following commands to run BERTQPP to estimate the retrieval quality of ConvDR on the test set of OR-QuAC:
```bash
python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3 \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 1  \
--interval 1000 \
--mode inference
```

#### qppBERTPL on OR-QuAC
Likewise, use the following command to train qppBERTPL to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC in terms of nDCG@3:
```bash
python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-train.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-train.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-train.actual-performance-run-manual-bm25-1000.json \
--epoch_num 1  \
--interval 1000
```
The training process would produce checkpoints, which are stored in `./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/`. 
Note that qppBERTPL does not need to be assigned a target metric. 

After training, use the following commands to run qppBERTPL to estimate the retrieval quality of T5-based, QuReTeC-based and human-written query rewrites+BM25 on the test set of OR-QuAC:
```bash
python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-T5-Q-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-QuReTeC-Q-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-manual-bm25-1000.json \
--epoch_num 1  \
--interval 1000 \
--mode inference
```
The output files of qppBERTPL would be saved in the path `./output/post-retrieval/`.

Use the following commands to run qppBERTPL to estimate the retrieval quality of ConvDR on the test set of OR-QuAC:
```bash
python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-QuReTeC-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--epoch_num 1  \
--interval 1000 \
--mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_name or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--run_path ./datasets/or-quac/runs/or-quac-test.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--epoch_num 1  \
--interval 1000 \
--mode inference
```

#### Assessing BM25 by NQAQPP on CAsT-19
Use the following commands to train NQAQPP in the setting of estimating the retrieval quality of T5-based, QuReTeC-based and human-written query rewrites+BM25 on CAsT-19 in terms of nDCG@3:
```bash
python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-T5-Q-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-QuReTeC-Q-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference
```
The above commands would train NQAQPP from scratch using 5-fold cross-validation (with the `--cross_validate` flag on). 
`--warm_up_path` shows the path to the checkpoint pre-trained on the training set of OR-QuAC. 
Note that one can activate `--warm_up_path`, which would fine-turn the checkpoint of NQAQPP pre-trained on the training set of OR-QuAC (warm-up) instead of just learning from scratch. 
We found that fine-tuning the checkpoint pre-trained for one epoch on the training set of OR-QuAC gets better performance. 
Please make sure the target metric used during 5-fold cross-validation and the target metric used during pre-training on OR-QuAC are the same. 
For example, if warm up NQAQPP from `./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@100/1.pkl`, one should set `--target_metric` as `ndcg@100`.
After training, activate `--mode inference` and execute the above command again to conduct inference; The inference process produces predicted performance files, which would be saved in the path `./output/post-retrieval/`.

#### Assessing ConvDR by NQAQPP on CAsT-19
ConvDR has a specially-trained query encoder to encode raw utterances. 
QPP methods designed for ad-hoc search do not have a special module to understand raw utterances.
When estimating the retrieval quality of ConvDR, we consider three types of inputs to a QPP method to help the QPP method understand the current query, namely T5-based, QuReTeC-based and human-written query rewrites.
Similarly, use the following commands to train NQAQPP to estimate the retrieval quality of ConvDR on CAsT-19 in terms of nDCG@3:
```bash
python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference
```

#### Assessing BM25 by BERTQPP on CAsT-19
Similarly, use the following commands to train BERTQPP in the setting of estimating the retrieval quality of T5-based, QuReTeC-based and human-written query rewrites+BM25 on CAsT-19 in terms of nDCG@3:
```bash
python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-T5-Q-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-QuReTeC-Q-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference
```

#### Assessing ConvDR by BERTQPP on CAsT-19
Similarly, use the following commands to train BERTQPP in the setting of estimating the retrieval quality of ConvDR on CAsT-19 in terms of nDCG@3:
```bash
python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference
```

#### Assessing BM25 by qppBERTPL on CAsT-19
Similarly, use the following commands to train qppBERTPL in the setting of estimating the retrieval quality of T5-based, QuReTeC-based and human-written query rewrites+BM25 on CAsT-19 in terms of nDCG@3:
```bash
python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-T5-Q-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-QuReTeC-Q-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-manual-bm25-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference
```

#### Assessing ConvDR by qppBERTPL on CAsT-19
Similarly, use the following commands to train qppBERTPL in the setting of estimating the retrieval quality of ConvDR on CAsT-19 in terms of nDCG@3:
```bash
python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-19.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference
```

#### Assessing BM25 by NQAQPP on CAsT-20
```bash
python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-T5-QA-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-QuReTeC-QA-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-QuReTeC-QA-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference
```

#### Assessing ConvDR by NQAQPP on CAsT-20
```bash
python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference

python -u ./supervisedQPP/NQAQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-NQAQPP-ndcg@3/1.pkl \
# --mode inference
```

#### Assessing BM25 by BERTQPP on CAsT-20

```bash
python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-T5-QA-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-QuReTeC-QA-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-QuReTeC-QA-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-manual-bm25-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference
```

#### Assessing ConvDR by BERTQPP on CAsT-20
```bash
python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference

python -u ./supervisedQPP/BERTQPP/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--target_metric ndcg@3 \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-BERTQPP-ndcg@3/1 \
# --mode inference
```

#### Assessing BM25 by qppBERTPL on CAsT-20
```bash
python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-T5-QA-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-QuReTeC-QA-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-QuReTeC-QA-bm25-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-manual-bm25-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-manual-bm25-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference
```

#### Assessing ConvDR by qppBERTPL on CAsT-20
```bash
python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference

python -u ./supervisedQPP/qppBERTPL/main.py \
--checkpoint_path ./checkpoint/ \
--output_path ./output/post-retrieval/ \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--run_path ./datasets/cast-19-20/runs/cast-20.run-ConvDR-1000.txt \
--actual_performance_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--epoch_num 10  \
--interval 10  \
--cross_validate \
# --warm_up_path ./checkpoint/or-quac-train.manual-bm25-1000.manual-qppBERTPL-classification/1.pkl \
# --mode inference
```

### Evaluation for Post-retrieval QPP Methods
The following commands are about evaluating all post-retrieval QPP methods in terms of Pearson's 𝜌, Kendall's 𝜏, and Spearman's 𝜌 correlation coefficients.
#### CAsT-19
Use the following commands to evaluate QPP methods when they estimate the retrieval quality of T5-based query rewrites+BM25, QuReTeC-based query rewrites+BM25, human-written query rewrites+BM25 and ConvDR on CAsT-19:
```bash
python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-19.T5-Q-bm25-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3 # target_metric can be set to "ndcg@3", "ndcg@100" or "recall@100"

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-19.QuReTeC-Q-bm25-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-19.manual-bm25-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-manual-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-19.ConvDR-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-ConvDR-1000.json \
--target_metrics ndcg@3 
```
```target_metric``` here can be set to "ndcg@3", "ndcg@100" or "recall@100". 
The files showing the evaluation results would be saved in the path `./output/post-retrieval/`. 
Note that when evaluating regression-based supervised QPP methods, such as NQAQPP and BERTQPP, make sure the target metric set here is consistent with the target metric in terms of which regression-based supervised methods are trained to estimate the retrieval quality during training.

#### CAsT-20
Likewise, Use the following command to evaluate QPP methods when they estimate the retrieval quality of T5-based query rewrites+BM25, QuReTeC-based query rewrites+BM25, human-written query rewrites+BM25 and ConvDR on CAsT-20:
```bash
python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-20.T5-QA-bm25-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--target_metrics ndcg@3 

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-20.QuReTeC-QA-bm25-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-QuReTeC-QA-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-20.manual-bm25-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-manual-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/cast-20.ConvDR-1000.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-ConvDR-1000.json \
--target_metrics ndcg@3 
```
The files showing the evaluation results would be saved in the path `./output/post-retrieval/`. 

#### OR-QuAC
Likewise, use the following command to evaluate QPP methods when they estimate the retrieval quality of T5-based query rewrites+BM25, QuReTeC-based query rewrites+BM25, human-written query rewrites+BM25 and ConvDR on the test set of OR-QuAC:
```bash
python -u evaluation_QPP.py \
--pattern './output/post-retrieval/or-quac-test.T5-Q-bm25-1000.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/or-quac-test.QuReTeC-Q-bm25-1000.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-QuReTeC-Q-bm25-1000.json \
--target_metrics ndcg@3 

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/or-quac-test.manual-bm25-1000.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-manual-bm25-1000.json \
--target_metrics ndcg@3 

python -u evaluation_QPP.py \
--pattern './output/post-retrieval/or-quac-test.ConvDR-1000.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-ConvDR-1000.json \
--target_metrics ndcg@3
```
The files showing the evaluation results would be saved in the path `./output/post-retrieval/`. 

## Plots
We added reported plots and the code for box plots in our paper in the `plots` folder. 
Note that for running the commands below, you need to have all the runs downloaded and stored in the ```datasets``` directory as instructed above.
```bash
python plots/plot_box_cast-19.py # generates CAsT-19.png
python plots/plot_box_cast-20.py # generates CAsT-20.png 
python plots/plot_box_or-quac.py # generates QuAC.png
```