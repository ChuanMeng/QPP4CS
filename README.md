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
srun --time=99:00:00 -c16 --mem=250G  python -u unsupervisedQPP/preretrieval_qpp.py \
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
srun --time=99:00:00 -c16 --mem=250G  python -u unsupervisedQPP/preretrieval_qpp.py \
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
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt
```

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt
```

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
```

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode baselines \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
```

## Compute Perplexity

```bash
srun -p gpu --gres=gpu:1 --time=99:00:00 -c8 --mem=50G  python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt
--LM gpt2-xl
```

```bash
srun -p gpu --gres=gpu:1 --time=99:00:00 -c8 --mem=50G  python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt
--LM gpt2-xl
```

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/or-quac/queries/or-quac-dev.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
--LM gpt2-xl
```

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode ppl \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
--LM gpt2-xl
```


## PPL-QPP

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-19.queries-T5-Q.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-19.qrels.txt \
--LM gpt2-xl \
--qpp_names VAR-std-sum  
```

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/cast-19-20/queries/cast-20.queries-T5-QA.tsv \
--index_path ./datasets/cast-19-20/index \
--qrels_path ./datasets/cast-19-20/qrels/cast-20.qrels.txt \
--LM gpt2-xl \
--qpp_names SCQ-avg 
```

```bash
srun --time=99:00:00 -c16 --mem=250G python -u unsupervisedQPP/preretrieval_qpp.py \
--mode PPL-QPP \
--query_path ./datasets/or-quac/queries/or-quac-test.queries-T5-Q.tsv \
--index_path ./datasets/or-quac/index \
--qrels_path ./datasets/or-quac/qrels/or-quac.qrels.txt
--LM gpt2-xl
--qpp_names VAR-std-sum
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
--target_metrics ndcg@3 ndcg@100
```

```bash
python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/or-quac-dev.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-dev.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3 ndcg@100
```

```bash
python -u evaluation_QPP.py \
--pattern './output/pre-retrieval/or-quac-test.*' \
--ap_path ./datasets/or-quac/actual_performance/or-quac-test.actual-performance-run-T5-Q-bm25-1000.json \
--target_metrics ndcg@3 ndcg@100
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

