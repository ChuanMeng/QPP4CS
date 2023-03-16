# QPP4CS

This is the repository for the paper entitled **Performance Prediction for Conversational Search Using Perplexities of Query Rewrites**.

This repository allows the replication of all results reported in the paper.
There are four steps:
- [Precomputation](#Precomputation): some of the pre-retrieval QPP methods (VAR and PMI) need precomputation and
- [Baselines](#Baselines)
- [Perplexity](#Perplexity)
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
## Baselines
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

## Perplexity

```bash


```

## PPL-QPP

```bash


```
