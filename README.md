# Query Performance Prediction for Conversational Search (QPP4CS) 
![visitors](https://visitor-badge.glitch.me/badge?page_id=AGI-Edgerunners/QPP4CS)
 
This is the repository for the reproducibility paper entitled [Query Performance Prediction: From Ad-hoc to Conversational Search]() and the workshop paper entitled [Performance Prediction for Conversational Search Using Perplexities of Query Rewrites](http://ceur-ws.org/Vol-3366/#paper-05).

Please cite our papers if you think this repository is helpful: 
```
@inproceedings{meng2023query,
 author = {Meng, Chuan and Arabzadeh, Negar and Aliannejadi, Mohammad and de Rijke, Maarten},
 title = {Query Performance Prediction: From Ad-hoc to Conversational Search},
 booktitle = {SIGIR 2023: The 46th international ACM SIGIR Conference on Research and Development in Information Retrieval},
 year = {2023},
}

@inproceedings{meng2023Performance,
 author = {Meng, Chuan and Aliannejadi, Mohammad and de Rijke, Maarten},
 title = {Performance Prediction for Conversational Search Using Perplexities of Query Rewrites},
 booktitle = {Proceedings of the The QPP++ 2023: Query Performance Prediction and Its Evaluation in New Tasks Workshop co-located with The 45th European Conference on Information Retrieval},
 year = {2023},
 pages = {25--28},
 url = {http://ceur-ws.org/Vol-3366/#paper-05},
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


## Prerequisites
We recommend running all the things in a Linux environment. 
Please create a conda environment with all required packages, and activate the environment by the following commands:
```
$ conda env create -f environment.yaml
$ conda activate QPP-CS
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
> For ease of use, we already uploaded the predicted performance files for all QPP methods reported in our paper. See [here](./predicted_performance_in_tables). 

### Pre-retrieval QPP methods
#### Precomputation
Some of the pre-retrieval QPP methods (VAR and PMI) would take a very long time to run. In order to reduce the time consumption, we first conduct precomputation on the two collections. 
The files of precomputation would be saved in the folder `./output/pre-retrieval/`. 
CAsT-19 and CAsT-20 share the same collection. Run the following command to do precomputation on the shared collection.
```bash
python -u preretrieval_qpp.py \
--mode precomputation \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--query_path_2 ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index
```
Run the following command to do precomputation on the collection of OR-QUAC.
```bash
python -u preretrieval_qpp.py \
--mode precomputation \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--query_path_2 ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index
```
#### Run Baselines
Run the following commands to run baselines on the CAsT-19, CAsT-20 and the test set of the OR-QUAC datasets:
```bash
python -u preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt

python -u preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt

python -u preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
```
The output files of baselines would be saved in the folder `./output/pre-retrieval/`. 

### Perplexity-based pre-retrieval QPP framework
#### Compute Perplexity

Run the following commands to compute the perplexities of query rewrites on the CAsT-19, CAsT-20 and the test set of the OR-QUAC datasets:
```bash
python -u preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--LM gpt2-xl

python -u preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--LM gpt2-xl

python -u preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--LM gpt2-xl
```
The output files would be saved in the folder `./output/pre-retrieval/**`. 

#### Run PPL-QPP
Run the following commands to run PPL-QPP on the CAsT-19, CAsT-20 and the test set of the OR-QUAC datasets:
```bash
python -u preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--LM gpt2-xl \
--qpp_names VAR-std-sum \
--alpha 0.1

python -u preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--LM gpt2-xl \
--qpp_names SCQ-avg \
--alpha 0.2

python -u preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--LM gpt2-xl \
--qpp_names VAR-std-sum \
--alpha 0

```
The output files of PPL-QPP would be saved in the folder `./output/pre-retrieval/**`.

### Evaluation for pre-retrieval QPP methods
Run the following commands to evaluate all pre-retrieval QPP methods and PPL-QPP in terms of Pearson, Kendall, and Spearman correlation coefficients, for estimating the retrieval quality of T5+BM25:
```bash
python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/cast-19.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/cast-20.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/or-quac-test.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3
```
The files showing the evaluation results would be saved in the folder `./output/pre-retrieval/**`.

### Post-retrieval unsupervised QPP methods

### Post-retrieval supervised QPP methods

### Evaluation for post-retrieval QPP methods


### Supervised-QPP-methods
We elaborate on how to replicate the result of a supervised QPP method under a specific setting in our paper. 
We consider three state-of-the-art supervised QPP methods, namely [NQAQPP](https://dl.acm.org/doi/abs/10.1145/3341981.3344249), [BERTQPP](https://dl.acm.org/doi/abs/10.1145/3459637.3482063) and [qppBERTPL](https://dl.acm.org/doi/abs/10.1145/3477495.3531821).
Note that we recommend using GPU to execute the following commands.

#### CAsT-19 and CAsT-20 (5-fold cross-validation)
We conduct 5-fold cross-validation on CAsT-19 or CAsT-20.

We show how to set up a model using an example of NQAQPP in the setting of estimating the retrieval quality of BM25 with T5-based query rewrites on CAsT-19 in terms of nDCG@3.
Use the following commands to set up NQAQPP in the setting of estimating the retrieval quality of BM25 with T5-based query rewrites on CAsT-19 in terms of nDCG@3:
```bash
export MODEL_TYPE=NQAQPP # can be set to "NQAQPP", "BERTQPP" or "qppBERTPL"
export DATASET_CLASS=cast-19-20 # can be set to "cast-19-20" or "or-quac"
export DATASET_NAME=cast-19 # can be set to "cast-19" or "cast-20" when DATASET_CLASS is "cast-19-20"; can be set to "or-quac-train", "or-quac-dev" and "or-quac-test" when DATASET_CLASS is "or-quac"
export QUERY_TYPE=T5-Q # can be set to "T5-Q", "QuReTeC-Q" or "manual" when DATASET_NAME is "cast-19", "or-quac-dev" or "or-quac-test"; can be set to "T5-QA", "QuReTeC-QA" and "manual" when DATASET_NAME is "cast-20"; always set it as "manual" when DATASET_NAME is "or-quac-train"
export RETRIEVER=T5-Q-bm25 # can be set to "T5-Q-bm25", "QuReTeC-Q-bm25", "manual-bm25" or "ConvDR" when DATASET_NAME is "cast-19", "or-quac-dev" or "or-quac-test"; can be set to "T5-QA-bm25", "QuReTeC-QA-bm25", "manual-bm25" or "ConvDR" when DATASET_NAME is "cast-20"; always set it as "manual-bm25" when DATASET_NAME is "or-quac-train"
export TARGET_METRIC=ndcg@3 # can be set to "ndcg@3", "ndcg@100" or "recall@100"; This variable would not impact qppBERTPL during training.

export MODEL_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${TARGET_METRIC}-${RETRIEVER} # remove `${TARGET_METRIC}` and get {DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${RETRIEVER} when running qppBERTPL that is independent to the target metric
export OUTPUT_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${TARGET_METRIC}-${RETRIEVER} # remove `${TARGET_METRIC}` and get ${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${RETRIEVER} when running qppBERTPL that is independent to the target metric
export QUERY_PATH=./datasets/${DATASET_CLASS}/queries/${DATASET_NAME}.queries-${QUERY_TYPE}.tsv
export RUN_PATH=./datasets/${DATASET_CLASS}/runs/${DATASET_NAME}.run-${RETRIEVER}-1000.txt
export ACTUAL_PERFORMANCE_PATH=./datasets/${DATASET_CLASS}/actual_performance/${DATASET_NAME}.actual-performance-run-${RETRIEVER}-1000.json
export INDEX_PATH=./datasets/${DATASET_CLASS}/index
export QRELS_PATH=./datasets/${DATASET_CLASS}/qrels/${DATASET_NAME}.qrels.txt
export EPOCH_NUM=10 
```
We mainly modify the following variables and one can change the setting to any case in our paper by modifying them: 
- `MODEL_TYPE` means the supervised QPP method to be used, which can be set to "NQAQPP", "BERTQPP" or "qppBERTPL".
- `DATASET_CLASS` means the category of the dataset to be used, which can be set to "cast-19-20" or "or-quac".
- `DATASET_NAME` means the specific dataset on which training and inference will be conducted, which can be set to "cast-19" or "cast-20" if `DATASET_CLASS` is set to "cast-19-20", and which can be set to "or-quac-train", "or-quac-dev" and "or-quac-test" if `DATASET_CLASS` is set to "or-quac".
- `QUERY_TYPE` means the query rewrite type to be fed into the QPP method, which can be set to "T5-Q", "QuReTeC-Q" or "manual" when `DATASET_NAME` is "cast-19", "or-quac-dev" or "or-quac-test", and can be set to "T5-QA", "QuReTeC-QA" and "manual" when `DATASET_NAME` is "cast-20". We always set `QUERY_TYPE` as "manual" when `DATASET_NAME` is "or-quac-train".
- `RETRIEVER` means the retriever to be evaluated by the QPP method, which can be set to "T5-Q-bm25", "QuReTeC-Q-bm25", "manual-bm25" or "ConvDR" when `DATASET_NAME` is "cast-19", "or-quac-dev" or "or-quac-test", and can be set to "T5-QA-bm25", "QuReTeC-QA-bm25", "manual-bm25" or "ConvDR" when `DATASET_NAME` is "cast-20". We always set `RETRIEVER` as "manual-bm25" when `DATASET_NAME` is "or-quac-train". Note that please always make sure the QPP method and BM25 use the same query rewrite type when predicting the retrieval quality of BM25.
- `TARGET_METRIC` means the IR metric in terms of which regression-based models (e.g., NQAQPP, BERTQPP) will learn to estimate the retrieval quality during training, which can be set to "ndcg@3", "ndcg@100" or "recall@100". Note that this variable will not influence qppBERTPL because it is a classification-based model and does not learn to approximate scores of a specific IR metric. Thus please remove `${TARGET_METRIC}` in `MODEL_NAME` and `OUTPUT_NAME` when running qppBERTP.

Use the following command to train NQAQPP using 5-fold cross-validation on CAsT-19 (with the `--cross_validate` flag on):
```
python -u supervisedQPP/${MODEL_TYPE}/main.py \
--model_name ${MODEL_NAME} \
--output_name ${OUTPUT_NAME} \
--dataset ${DATASET_NAME} \
--query_path ${QUERY_PATH} \
--index_path ${INDEX_PATH} \
--qrels_path ${QRELS_PATH} \
--run_path ${RUN_PATH} \
--actual_performance_path ${ACTUAL_PERFORMANCE_PATH} \
--target_metric ${TARGET_METRIC} \
--epoch_num ${EPOCH_NUM} \
--cross_validate \
--mode training # can be changed to "inference" after finishing the training
```
The training process will produce checkpoints over all epochs, which are stored in `./output/${MODEL_NAME}/checkpoint/`, namely `./output/cast-19.T5-Q-NQAQPP-ndcg@3-T5-Q-bm25/checkpoint/`.

Next, set the flag `--mode` as `inference` and execute the above command again to conduct inference.
The inference process produces predicted performance files over all epochs, which are stored in `./output/${MODEL_NAME}/`, namely `./output/cast-19.T5-Q-NQAQPP-ndcg@3-T5-Q-bm25/`.

Use the following command to evaluate NQAQPP on CAsT-19 in terms of Pearson's 𝜌, Kendall's 𝜏, and Spearman's 𝜌 correlation coefficients:
```
python -u evaluation_QPP.py --ap_path ${ACTUAL_PERFORMANCE_PATH} --pp_path output/${MODEL_NAME}/${OUTPUT_NAME} --epoch_num ${EPOCH_NUM} --target_metric ${TARGET_METRIC}
```
The above command will generate a result file (`${OUTPUT_NAME}.${TARGET_METRIC}`, namely **cast-19.T5-Q-NQAQPP-ndcg@3-T5-Q-bm25.ndcg@3**) showing the correlation scores for all epochs, which is stored in `./output/${MODEL_NAME}/`, namely `./output/cast-19.T5-Q-NQAQPP-ndcg@3-T5-Q-bm25/`.

#### OR-QuAC
We first train a QPP model on the training set of OR-QuAC, and then conduct inference on the development set and test set of OR-QuAC.
Note that we always train a QPP model to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC.
It is because [the T5 rewriter](https://huggingface.co/castorini/t5-base-canard) and [QuReTeC](https://github.com/nickvosk/sigir2020-query-resolution) we use in this paper are trained over the queries in the training set of OR-QuAC.
Thus it is unreasonable to run them on the training set of OR-QuAC.
We pick the best checkpoint based on the development set in terms of Pearson's correlation scores.
We still use an example of NQAQPP in the setting of estimating the retrieval quality of BM25 with T5-based query rewrites on OR-QuAC in terms of nDCG@3.

First we set up and train NQAQPP to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC in terms of nDCG@3:
```bash
export MODEL_TYPE=NQAQPP 
export DATASET_CLASS=or-quac
export DATASET_NAME=or-quac-train
export QUERY_TYPE=manual # always set it as "manual" when DATASET_NAME is "or-quac-train" 
export RETRIEVER=manual-bm25 # always set it as "manual-bm25" when DATASET_NAME is "or-quac-train"
export TARGET_METRIC=ndcg@3 

export MODEL_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${TARGET_METRIC}-${RETRIEVER}
export OUTPUT_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${TARGET_METRIC}-${RETRIEVER}
export QUERY_PATH=./datasets/${DATASET_CLASS}/queries/${DATASET_NAME}.queries-${QUERY_TYPE}.tsv
export RUN_PATH=./datasets/${DATASET_CLASS}/runs/${DATASET_NAME}.run-${RETRIEVER}-1000.txt
export ACTUAL_PERFORMANCE_PATH=./datasets/${DATASET_CLASS}/actual_performance/${DATASET_NAME}.actual-performance-run-${RETRIEVER}-1000.json
export INDEX_PATH=./datasets/${DATASET_CLASS}/index
export QRELS_PATH=./datasets/${DATASET_CLASS}/qrels/${DATASET_CLASS}.qrels.txt
export EPOCH_NUM=10 

python -u supervisedQPP/${MODEL_TYPE}/main.py \
--model_name ${MODEL_NAME} \
--output_name ${OUTPUT_NAME} \
--dataset ${DATASET_NAME} \
--query_path ${QUERY_PATH} \
--index_path ${INDEX_PATH} \
--qrels_path ${QRELS_PATH} \
--run_path ${RUN_PATH} \
--actual_performance_path ${ACTUAL_PERFORMANCE_PATH} \
--target_metric ${TARGET_METRIC} \
--epoch_num ${EPOCH_NUM} \
--mode training 
```
The training process would produce checkpoints over all epochs, which are stored in `./output/${MODEL_NAME}/checkpoint/`, namely `./output/or-quac-train.manual-NQAQPP-ndcg@3-manual-bm25/checkpoint/`.

Set up variables and conduct inference on the development set of OR-QuAC:
```bash
export MODEL_TYPE=NQAQPP 
export DATASET_CLASS=or-quac
export DATASET_NAME=or-quac-dev 
export QUERY_TYPE=T5-Q
export RETRIEVER=T5-Q-bm25
export TARGET_METRIC=ndcg@3 

export MODEL_NAME=or-quac-train.manual-${MODEL_TYPE}-ndcg@3-manual-bm25
export OUTPUT_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${TARGET_METRIC}-${RETRIEVER}
export QUERY_PATH=./datasets/${DATASET_CLASS}/queries/${DATASET_NAME}.queries-${QUERY_TYPE}.tsv
export RUN_PATH=./datasets/${DATASET_CLASS}/runs/${DATASET_NAME}.run-${RETRIEVER}-1000.txt
export ACTUAL_PERFORMANCE_PATH=./datasets/${DATASET_CLASS}/actual_performance/${DATASET_NAME}.actual-performance-run-${RETRIEVER}-1000.json
export INDEX_PATH=./datasets/${DATASET_CLASS}/index
export QRELS_PATH=./datasets/${DATASET_CLASS}/qrels/${DATASET_CLASS}.qrels.txt
export EPOCH_NUM=10 

python -u supervisedQPP/${MODEL_TYPE}/main.py \
--model_name ${MODEL_NAME} \
--output_name ${OUTPUT_NAME} \
--dataset ${DATASET_NAME} \
--query_path ${QUERY_PATH} \
--index_path ${INDEX_PATH} \
--qrels_path ${QRELS_PATH} \
--run_path ${RUN_PATH} \
--actual_performance_path ${ACTUAL_PERFORMANCE_PATH} \
--target_metric ${TARGET_METRIC} \
--epoch_num ${EPOCH_NUM} \
--mode inference
```
The inference process produces predicted performance files over all epochs on the development set of OR-QuAC, which are stored in `./output/${MODEL_NAME}/`, namely `./output/or-quac-train.manual-NQAQPP-ndcg@3-manual-bm25`.
Use the following command to evaluate NQAQPP on the development set of OR-QuAC in terms of Pearson's 𝜌, Kendall's 𝜏, and Spearman's 𝜌 correlation coefficients:
```bash
python -u evaluation_QPP.py --ap_path ${ACTUAL_PERFORMANCE_PATH} --pp_path output/${MODEL_NAME}/${OUTPUT_NAME} --epoch_num ${EPOCH_NUM} --target_metric ${TARGET_METRIC}
```
The above command will generate a result file (`${OUTPUT_NAME}.${TARGET_METRIC}`, namely **or-quac-dev.T5-Q-NQAQPP-ndcg@3-T5-Q-bm25.ndcg@3**) showing the correlation scores over all epochs on the development set of OR-QuAC, which is stored in `./output/${MODEL_NAME}/`, namely `./output/or-quac-train.manual-NQAQPP-ndcg@3-manual-bm25/`.

Set up variables and conduct inference on the test set of OR-QuAC:
```bash
export MODEL_TYPE=NQAQPP 
export DATASET_CLASS=or-quac
export DATASET_NAME=or-quac-test 
export QUERY_TYPE=T5-Q
export RETRIEVER=T5-Q-bm25
export TARGET_METRIC=ndcg@3 

export MODEL_NAME=or-quac-train.manual-${MODEL_TYPE}-ndcg@3-manual-bm25
export OUTPUT_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${TARGET_METRIC}-${RETRIEVER}
export QUERY_PATH=./datasets/${DATASET_CLASS}/queries/${DATASET_NAME}.queries-${QUERY_TYPE}.tsv
export RUN_PATH=./datasets/${DATASET_CLASS}/runs/${DATASET_NAME}.run-${RETRIEVER}-1000.txt
export ACTUAL_PERFORMANCE_PATH=./datasets/${DATASET_CLASS}/actual_performance/${DATASET_NAME}.actual-performance-run-${RETRIEVER}-1000.json
export INDEX_PATH=./datasets/${DATASET_CLASS}/index
export QRELS_PATH=./datasets/${DATASET_CLASS}/qrels/${DATASET_CLASS}.qrels.txt
export EPOCH_NUM=10 

python -u supervisedQPP/${MODEL_TYPE}/main.py \
--model_name ${MODEL_NAME} \
--output_name ${OUTPUT_NAME} \
--dataset ${DATASET_NAME} \
--query_path ${QUERY_PATH} \
--index_path ${INDEX_PATH} \
--qrels_path ${QRELS_PATH} \
--run_path ${RUN_PATH} \
--actual_performance_path ${ACTUAL_PERFORMANCE_PATH} \
--target_metric ${TARGET_METRIC} \
--epoch_num ${EPOCH_NUM} \
--mode inference
```
The inference process produces predicted performance files over all epochs on the test set of OR-QuAC, which are stored in `./output/${MODEL_NAME}/`, namely `./output/or-quac-train.manual-NQAQPP-ndcg@3-manual-bm25`.
Use the following command to evaluate NQAQPP on the test set of OR-QuAC in terms of Pearson's 𝜌, Kendall's 𝜏, and Spearman's 𝜌 correlation coefficients:
```bash
python -u evaluation_QPP.py --ap_path ${ACTUAL_PERFORMANCE_PATH} --pp_path output/${MODEL_NAME}/${OUTPUT_NAME} --epoch_num ${EPOCH_NUM} --target_metric ${TARGET_METRIC}
```
The above command will generate a result file (`${OUTPUT_NAME}.${TARGET_METRIC}`, namely **or-quac-test.T5-Q-NQAQPP-ndcg@3-T5-Q-bm25.ndcg@3**) showing the correlation scores over all epochs on the test set of OR-QuAC, which is stored in `./output/${MODEL_NAME}/`, namely `./output/or-quac-train.manual-NQAQPP-ndcg@3-manual-bm25/`.

#### CAsT-19 and CAsT-20 (Warm-Up) 
We consider a warm-up setting, where we fine-turn the model, pre-trained on the training set of OR-QuAC (warm-up), using 5-fold cross-validation on CAsT-19 or CAsT-20.
We still use an example of NQAQPP; it estimates the retrieval quality of BM25 with T5-based query rewrites on CAsT-19 in terms of nDCG@3.
We first pre-train NQAQPP to estimate the retrieval quality of BM25 with human-rewritten queries on the training set of OR-QuAC in terms of nDCG@3; the training process is the same as  the training process in the above section.
Thus we directly assume that we already finish the pre-training of NQAQPP on the training set of OR-QuAC (warm-up), and the pre-trained checkpoints are stored in `./output/${MODEL_NAME}/checkpoint/`, namely `./output/or-quac-train.manual-NQAQPP-ndcg@3-manual-bm25/checkpoint/`.

We found that fine-tuning the checkpoint pre-trained for one epoch on the training set of OR-QuAC gets better performance.
So we fine-tune NQAQPP, pre-trained for one epoch on the training set of OR-QuAC, on CAsT-19 using 5-fold cross-validation (with the `--cross_validate` flag on):
```bash
export MODEL_TYPE=NQAQPP 
export DATASET_CLASS=cast-19-20
export DATASET_NAME=cast-19 
export QUERY_TYPE=T5-Q 
export RETRIEVER=T5-Q-bm25 
export TARGET_METRIC=ndcg@3

export MODEL_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-warm-up-${TARGET_METRIC}-${RETRIEVER}
export OUTPUT_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-warm-up-${TARGET_METRIC}-${RETRIEVER}
export QUERY_PATH=./datasets/${DATASET_CLASS}/queries/${DATASET_NAME}.queries-${QUERY_TYPE}.tsv
export RUN_PATH=./datasets/${DATASET_CLASS}/runs/${DATASET_NAME}.run-${RETRIEVER}-1000.txt
export ACTUAL_PERFORMANCE_PATH=./datasets/${DATASET_CLASS}/actual_performance/${DATASET_NAME}.actual-performance-run-${RETRIEVER}-1000.json
export INDEX_PATH=./datasets/${DATASET_CLASS}/index
export QRELS_PATH=./datasets/${DATASET_CLASS}/qrels/${DATASET_NAME}.qrels.txt
export EPOCH_NUM=10 

export WARM_UP_PATH=./output/or-quac-train.manual-${MODEL_TYPE}-ndcg@3-manual-bm25/checkpoint/1.pkl 
# "1.pkl" means the checkpoint pre-trained for one epoch on the training set of OR-QuAC. 
# please use ".../checkpoint/1.pkl" for NQAQPP or qppBERTPL but use ".../checkpoint/1" for BERTQPP

python -u supervisedQPP/${MODEL_TYPE}/main.py \
--model_name ${MODEL_NAME} \
--output_name ${OUTPUT_NAME} \
--dataset ${DATASET_NAME} \
--query_path ${QUERY_PATH} \
--index_path ${INDEX_PATH} \
--qrels_path ${QRELS_PATH} \
--run_path ${RUN_PATH} \
--actual_performance_path ${ACTUAL_PERFORMANCE_PATH} \
--target_metric ${TARGET_METRIC} \
--epoch_num ${EPOCH_NUM} \
--warm_up_path ${WARM_UP_PATH} \
--cross_validate \
--mode training # can be changed to "inference" after finishing the training
```
`WARM_UP_PATH` shows the path to the checkpoint pre-trained on the training set of OR-QuAC. 
"1.pkl" means the checkpoint pre-trained for one epoch on the training set of OR-QuAC.
Note that please use ".../checkpoint/1.pkl" for NQAQPP or qppBERTPL but use ".../checkpoint/1" for BERTQPP.

Next, set the flag `--mode` as `inference` and execute the above command again to conduct inference.
The inference process produces predicted performance files over all epochs on CAsT-19, which are stored in `./output/${MODEL_NAME}/`, namely `./output/cast-19.T5-Q-NQAQPP-warm-up-ndcg@3-T5-Q-bm25/`.

Finally, use the following command to evaluate NQAQPP on CAsT-19 in terms of Pearson's 𝜌, Kendall's 𝜏, and Spearman's 𝜌 correlation coefficients:
```bash
python -u evaluation_QPP.py --ap_path ${ACTUAL_PERFORMANCE_PATH} --pp_path output/${MODEL_NAME}/${OUTPUT_NAME} --epoch_num ${EPOCH_NUM} --target_metric ${TARGET_METRIC}
```
The above command will generate a result file (`${OUTPUT_NAME}.${TARGET_METRIC}`, namely **cast-19.T5-Q-NQAQPP-warm-up-ndcg@3-T5-Q-bm25.ndcg@3**) showing the correlation scores for all epochs, which is stored in `./output/${MODEL_NAME}/`, namely `./output/cast-19.T5-Q-NQAQPP-warm-up-ndcg@3-T5-Q-bm25/`.

### Unsupervised QPP methods
#### NQC, SMV, WIG, σ<sub>k</sub>, σ<sub>%</sub>
The following script allow to run ```NQC```, ```WIG```, ```SMV```, ```sigma_K```, ```sigma_persentage``` QPP methods:

In order to run the unsupervised QPP methods we covered in the paper, you can use the ```unsupervised_qpp.py``` script as follows:
```bash
python unsupervisedQPP/unsupervised_qpp.py \
--run_file {path_to_your_run_file} \
--query_file {path_to_your_query_file} \
--k {cut_off depth} \
--qpp_method {any of the QPP methods reported in the paper including nqc, wig, smv, simga_k, sigma_percentage,...}\
--output {path_to_output_file}
```
The output file would include ```qid \t predicted performance``` per line. Note that some of the QPP methods, such as clarity, would take longer to run. 
You can find the predicted performance files for all unsupervised QPP methods reported in our paper from [here](./predicted_performance_in_tables).

For instance, assuming you have followed the previous steps and you have the run files stored in the ```datasets``` folder, the following command could predict the performance of BM25 with the T5 query rewriter on the cast-19 dataset using the NQC method when using top-100 retrieved documents:
```bash
export MODEL_TYPE=nqc # can be set to "nqc", "smv", 'wig', 'sigma_k', 'sigma_percentage', ...
export DATASET_CLASS=cast-19-20
export DATASET_NAME=cast-19 
export QUERY_TYPE=T5-Q 
export RETRIEVER=T5-Q-bm25 
export OUTPUT_NAME=${DATASET_NAME}.${QUERY_TYPE}-${MODEL_TYPE}-${RETRIEVER} 
export QUERY_PATH=./datasets/${DATASET_CLASS}/queries/${DATASET_NAME}.queries-${QUERY_TYPE}.tsv
export RUN_PATH=./datasets/${DATASET_CLASS}/runs/${DATASET_NAME}.run-${RETRIEVER}-1000.txt
export ACTUAL_PERFORMANCE_PATH=./datasets/${DATASET_CLASS}/actual_performance/${DATASET_NAME}.actual-performance-run-${RETRIEVER}-1000.json
```
```bash
python unsupervisedQPP/unsupervised_qpp.py \
--run_file ${RUN_PATH} \
--query_file ${QUERY_PATH}  \
--k 100 \
--qpp_method ${MODEL_TYPE} \
--output ./output/${MODEL_TYPE}/${OUTPUT_NAME}
```
The above command will produce a predicted performance file (`${OUTPUT_NAME}`, namely **cast-19.T5-Q-nqc-T5-Q-bm25**) that is stored in `./output/${MODEL_TYPE}/`, namely `./output/nqc/`.

#### QF
The following script allow to run ```QF``` QPP method which relies on psuedo relevance feedback. To run these metric, first we need to conduct the retrieval with rm3 feature enabled as follows :

For CAsT-19 :
```bash
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-19.run-T5-Q-bm25-rm3-1000.txt --bm25 --rm3  --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-19.run-QuReTeC-Q-bm25-rm3-1000.txt --bm25  --rm3 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-19.queries-manual.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-19.run-manual-bm25-rm3-1000.txt --bm25  --rm3  --hits 1000 --threads 16 --batch-size 64
```

For CAsT-20 :
```bash
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-20.run-T5-QA-bm25-1000.txt --bm25 --rm3 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-20.run-QuReTeC-QA-bm25-1000.txt --bm25 --rm3 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/cast-19-20/queries/cast-20.queries-manual.tsv  --index datasets/cast-19-20/index --output datasets/cast-19-20/runs/cast-20.run-manual-bm25-1000.txt --bm25 --rm3 --hits 1000 --threads 16 --batch-size 64 
```

For OR-QuAC:
```bash
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-dev.run-T5-Q-bm25-rm3-1000.txt --bm25 --rm3 --hits 1000 --threads 16 --batch-size 64
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-dev.queries-QuReTeC-Q.tsv  --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-dev.run-QuReTeC-Q-bm25-rm3-1000.txt --bm25 --rm3 --hits 1000 --threads 16 --batch-size 64 
python -m pyserini.search.lucene --topics datasets/or-quac/queries/or-quac-dev.queries-manual.tsv --index datasets/or-quac/index --output datasets/or-quac/runs/or-quac-dev.run-manual-bm25-rm3-1000.txt --bm25 --rm3 --hits 1000 --threads 16 --batch-size 64
```

The input for  QF methods includes the original run_file (using BM25) as well as the runfile we just created using BM25+RM3.  As such, we run these unsupervised QPP methods as follows:
```bash
export RUN_RM3_PATH=./datasets/${DATASET_CLASS}/runs/${DATASET_NAME}.run-${RETRIEVER}-rm3-1000.txt
export MODEL_TYPE=qf
```
```bash
python unsupervisedQPP/unsupervised_qpp.py \
--run_file ${RUN_PATH} \
--run_file_prf ${RUN_RM3_PATH} \
--query_file ${QUERY_PATH}  \
--k 100 \
--qpp_method  ${MODEL_TYPE}\
--output ./output/${MODEL_TYPE}/${OUTPUT_NAME}
```

#### UEF
The following script allow to run ```UEF``` QPP method which similar to QF also relies on psuedo relevance feedback. In other words, it requires the original run file as well as the retrieval results with psuedo relevance feedback. In addition, UEF requires one of the previous QPP methods prediuction result. In the paper, we reported UEF(NQC), as such, to run UEF, it is required to have the NQC results ready beforehand. Assuming we have the NQC predicted results, the following script would run UEF(NQC) QPP method:

```bash
export QPP_BASE=${DATASET_NAME}.${QUERY_TYPE}-nqc-${RETRIEVER} 
export MODEL_TYPE=uef
```

```bash
python unsupervisedQPP/unsupervised_qpp.py \
--run_file ${RUN_PATH} \
--run_file_prf ${RUN_RM3_PATH} \
--query_file ${QUERY_PATH}  \
--k 100 \
--qpp_method ${MODEL_TYPE}\
--qpp_base ${QPP_BASE} \
--output ./output/${MODEL_TYPE}/${OUTPUT_NAME}
```

#### Clarity
To calculate Clarity QPP method we are requires some term statiscs infdormation from the index. Therefore this method takes the index as one of its input. To run clarity QPP method, you can run:

```bash
export INDEX_PATH= datasets/cast-19-20/index #or  datasets/or-quac/index for or-quac dataset
export MODEL_TYPE=clarity
```

```bash
python unsupervisedQPP/unsupervised_qpp.py \
--run_file ${RUN_PATH} \
--query_file ${QUERY_PATH}  \
--index ${INDEX_PATH}
--k 100 \
--qpp_method ${MODEL_TYPE}\
--output ./output/${MODEL_TYPE}/${OUTPUT_NAME}
```

We note that unlike previouw metrics, clarity score is very slow and it has pretty high running time since it ierate over all the words in the corpus. Thus, running it with ```k=100``` could take long, especially on or-quac dataset which include high number of queries.


 #### Evaluation
Note that all unsupervised QPP methods are independent of a specific target metric. 
Similar to the supervised methods, we use the following command to evaluate unsupervised QPP methods in terms of Pearson's 𝜌, Kendall's 𝜏, and Spearman's 𝜌 correlation coefficients:
```bash
export TARGET_METRIC=ndcg@3
python -u evaluation_QPP.py --ap_path ${ACTUAL_PERFORMANCE_PATH} --pp_path output/${MODEL_TYPE}/${OUTPUT_NAME} --target_metric ${TARGET_METRIC}
```
```target_metric``` here can be set to "ndcg@3", "ndcg@100" or "recall@100".

## Plots
We added reported plots and the code for box plots in our paper in the `plots` folder. 
Note that for running the commands below, you need to have all the runs downloaded and stored in the ```datasets``` directory as instructed above.
```bash
python plots/plot_box_cast-19.py # generates CAsT-19.png
python plots/plot_box_cast-20.py # generates CAsT-20.png 
python plots/plot_box_or-quac.py # generates QuAC.png
```
