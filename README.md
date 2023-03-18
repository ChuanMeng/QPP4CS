# QPP4CS

This is the repository for the paper **Performance Prediction for Conversational Search Using Perplexities of Query Rewrites**.
In order to replicate the results reported in the paper, please follow four steps:
- [Precomputation](#Precomputation): some of the pre-retrieval QPP methods (VAR and PMI) need precomputation and
- [Run Baselines](#Run-Baselines)
- [Compute Perplexity](#Compute-Perplexity)
- [PPL-QPP](#PPL-QPP)


## Precomputation
```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode precomputation \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--query_path_2 ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index
```

```bash
python -u unsupervisedQPP/preretrieval_qpp.py \
--mode precomputation \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--query_path_2 ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
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
@inproceedings{chuan2023,
 author = {Meng, Chuan and Aliannejadi, Mohammad and de Rijke, Maarten},
 booktitle = {QPP++ 2023: Query Performance Prediction and Its Evaluation in New Tasks},
 title = {Performance Prediction for Conversational Search Using Perplexities of Query Rewrites},
 year = {2023}
}
```

## Question
Feel free to contact Chuan Meng (c.meng AT uva.nl) if you have any questions. 

