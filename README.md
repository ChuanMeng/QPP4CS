# QPP4CS

This is the repository for the paper entitled **Performance Prediction for Conversational Search Using Perplexities of Query Rewrites**.

This repository allows the replication of all results reported in the paper.
There are four steps:
- [Precomputation](#Precomputation): some of the pre-retrieval QPP methods (VAR and PMI) need precomputation and
- [Run Baselines](#Run-Baselines)
- [Compute Perplexity](#Compute-Perplexity)
- [PPL-QPP](#PPL-QPP)


## Precomputation
```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode precomputation \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--query_path_2 ./datasets/cast-19-20/queries/cast-19.queries-raw.tsv \
--query_path_3 ./datasets/cast-19-20/queries/cast-19.queries-manual.tsv \
--query_path_4 ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--query_path_5 ./datasets/cast-19-20/queries/cast-20.queries-raw.tsv \
--query_path_6 ./datasets/cast-19-20/queries/cast-20.queries-manual.tsv \
--index_path ./datasets/cast-19-20/index
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode precomputation \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--query_path_2 ./datasets/or-quac/queries/or-quac-test.queries-raw.tsv \
--query_path_3 ./datasets/or-quac/queries/or-quac-test.queries-manual.tsv \
--query_path_4 ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
--query_path_5 ./datasets/or-quac/queries/or-quac-dev.queries-raw.tsv \
--query_path_6 ./datasets/or-quac/queries/or-quac-dev.queries-manual.tsv \
--index_path ./datasets/or-quac/index
```
## Run Baselines
```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
```

## Compute Perplexity

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--LM gpt2-xl
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--LM gpt2-xl
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--LM gpt2-xl
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--LM gpt2-xl
```


## PPL-QPP

Alpha 

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--LM gpt2-xl \
--qpp_names VAR-std-sum \
--alpha 0.1
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--LM gpt2-xl \
--qpp_names SCQ-avg \
--alpha 0.2
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt \
--LM gpt2-xl \
--qpp_names VAR-std-sum \
--alpha 0

```


```bash
python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/cast-19.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-19.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3
```

```bash
python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/cast-20.*' \
--ap_path ./datasets/cast-19-20/actual_performance/cast-20.actual-performance-run-T5-QA-bm25-1000.json \
--target_metrics ndcg@3
```

```bash
python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/or-quac-test.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3
```


## Citation
Please cite our paper if you think this repository is helpful: 
```
@inproceedings{
xx
}
```

## Question
Feel free to contact Chuan Meng (c.meng AT uva.nl) if you have any questions. 

