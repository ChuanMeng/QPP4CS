# QPP4CS

This is the repository for the paper **Performance Prediction for Conversational Search Using Perplexities of Query Rewrites**.
In order to replicate the results reported in the paper, please follow the following five steps:
- [Prerequisites](#Prerequisites)
- [Precomputation](#Precomputation)
- [Run Baselines](#Run-Baselines)
- [Compute Perplexity](#Compute-Perplexity)
- [Run PPL-QPP](#Run-PPL-QPP)
- [Evaluation](#Evaluation) 

## Prerequisites
Install dependencies:  
```bash
pip install -r requirements.txt
```

## Precomputation
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
## Run Baselines
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

## Compute Perplexity

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

## Run PPL-QPP
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

## Run PPL-QPP
Lastly, run the following commands to evaluate all baselines and PPL-QPP in terms of Pearson, Kendall, and Spearman correlation coefficients:
```bash
python -u evaluation_qpp.py \
--pattern './output/pre-retrieval/cast-19.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_qpp.py \
--pattern './output/pre-retrieval/cast-20.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--target_metrics ndcg@3

python -u evaluation_qpp.py \
--pattern './output/pre-retrieval/or-quac-test.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3
```
The files showing the evaluation results would be saved in the folder `./output/pre-retrieval/**`.

## Citation
Please cite our paper if you think this repository is helpful: 
```
@inproceedings{chuan2023Performance,
 author = {Meng, Chuan and Aliannejadi, Mohammad and de Rijke, Maarten},
 booktitle = {QPP++ 2023: Query Performance Prediction and Its Evaluation in New Tasks},
 title = {Performance Prediction for Conversational Search Using Perplexities of Query Rewrites},
 year = {2023}
}
```

## Question
Feel free to contact Chuan Meng (c.meng AT uva.nl) if you have any questions. 

