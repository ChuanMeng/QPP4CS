import argparse
import json
from tqdm import tqdm
import os
from trec_car import read_data

def parse_sim_file(filename):
    """
    Reads the deduplicated documents file and stores the
    duplicate passage ids into a dictionary
    """
    sim_dict = {}
    lines = open(filename).readlines()
    for line in lines:
        data = line.strip().split(':')

        if len(data[1]) > 0:
            sim_docs = data[-1].split(',')
            for docs in sim_docs:
                sim_dict[docs] = 1

    return sim_dict

def ProcessCAsT():
    print("Processing CAsT-19 queries...")

    if not os.path.exists("datasets/cast-19-20/queries"):
        os.makedirs("datasets/cast-19-20/queries")

    with open("datasets/cast-19-20/raw/evaluation_topics_v1.0.json", "r") as fin:
        cast19_raw = json.load(fin)

    with open("datasets/cast-19-20/raw/evaluation_topics_annotated_resolved_v1.0.tsv", "r") as fin:
        cast19_manual = fin.readlines()

    w_cast19_raw = open("datasets/cast-19-20/queries/cast-19.queries-raw.tsv", "w")
    w_cast19_manual= open("datasets/cast-19-20/queries/cast-19.queries-manual.tsv", "w")

    for group in cast19_raw:
        topic_number, description, turn, title = str(group['number']), group.get('description', ''), group['turn'], group.get('title', '')
        queries = []
        for query in turn:
            query_number, raw_utterance = str(query['number']), query['raw_utterance']
            queries.append(raw_utterance)
            w_cast19_raw.write("{}_{}\t{}\n".format(topic_number, query_number, raw_utterance))  # 31_1 text
    w_cast19_raw.close()

    for line in cast19_manual:
        w_cast19_manual.write(line)
    w_cast19_manual.close()


    print("Processing CAsT-20 queries...")

    with open("datasets/cast-19-20/raw/2020_automatic_evaluation_topics_v1.0.json", "r") as f:
        cast20_raw = json.load(f)
    with open("datasets/cast-19-20/raw/2020_manual_evaluation_topics_v1.0.json", "r") as f:
        cast20_manual = json.load(f)

    w_cast20_raw = open("datasets/cast-19-20/queries/cast-20.queries-raw.tsv", "w")
    w_cast20_manual = open("datasets/cast-19-20/queries/cast-20.queries-manual.tsv", "w")


    for auto_topic, manual_topic in zip(cast20_raw, cast20_manual):
        topic_number = auto_topic["number"]
        assert topic_number == manual_topic["number"]

        auto_turns = auto_topic["turn"]
        manual_turns = manual_topic["turn"]

        assert len(auto_turns) == len(manual_turns)

        for auto_turn, manual_turn in zip(auto_turns, manual_turns):
            query_number = auto_turn["number"]
            raw = auto_turn["raw_utterance"]
            target = manual_turn["manual_rewritten_utterance"]

            w_cast20_raw.write(str(topic_number) + "_" + str(query_number) + "\t" + raw + "\n")
            w_cast20_manual.write(str(topic_number) + "_" + str(query_number) + "\t" + target + "\n")

    w_cast20_raw.close()
    w_cast20_manual.close()

    print("Processing CAsT-19 qrel...")

    if not os.path.exists("datasets/cast-19-20/qrels"):
        os.makedirs("datasets/cast-19-20/qrels")

    with open("datasets/cast-19-20/raw/2019qrels.txt", "r") as r:
        cast19_qrel = r.readlines()
    w_cast19_qrel = open("datasets/cast-19-20/qrels/cast-19.qrels.txt", "w")
    for line in cast19_qrel:
        w_cast19_qrel.write(line)
    w_cast19_qrel.close()

    print("Processing CAsT-20 qrel...")
    with open("datasets/cast-19-20/raw/2020qrels.txt", "r") as r:
        cast20_qrel = r.readlines()
    w_cast20_qrel = open("datasets/cast-19-20/qrels/cast-20.qrels.txt", "w")
    for line in cast20_qrel:
        w_cast20_qrel.write(line)
    w_cast20_qrel.close()


    print("Processing the collection shared by CAsT-19 and CAsT-20...")
    if not os.path.exists("datasets/cast-19-20/jsonl"):
        os.makedirs("datasets/cast-19-20/jsonl")

    sim_dict = parse_sim_file("datasets/cast-19-20/raw/duplicate_list_v1.0.txt")  # datasets/raw/duplicate_list_v1.0.txt  {"..":1,...}
    count = 0
    with open("datasets/cast-19-20/jsonl/cast-19-20.jsonl", "w") as f:
        print("Processing TREC-CAR...")
        for para in tqdm(read_data.iter_paragraphs(open("datasets/cast-19-20/raw/paragraphCorpus/dedup.articles-paragraphs.cbor", 'rb'))):  # the "rb" mode opens the file in binary format for reading
            car_id = "CAR_" + para.para_id
            text = para.get_text()
            text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(json.dumps({"id": car_id, "contents": text}) + "\n")
            count += 1

        print("Processing MS MARCO...")
        removed = 0
        with open("datasets/cast-19-20/raw/msmarco.tsv", "r") as m:
            for line in tqdm(m):
                marco_id, text = line.strip().split("\t")
                marco_id = "MARCO_" + marco_id
                text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ")

                if marco_id in sim_dict:
                    removed += 1
                    continue

                f.write(json.dumps({"id": marco_id,"contents": text}) + "\n")
                count += 1

        print("Removed " + str(removed) + " passages")
        print(f"The number of passsages: {count}")
    return None


def ProcessORQuAC():
    print("Processing OR-QuAC queries...")

    if not os.path.exists("datasets/or-quac/queries"):
        os.makedirs("datasets/or-quac/queries")

    target_names = ['train', 'dev', 'test']
    idx = 0
    for target_name in target_names:
        print(f"Processing {target_name}.txt")
        target = os.path.join("datasets/or-quac/raw/", f"{target_name}.txt")
        queries_manual = os.path.join("datasets/or-quac/queries/", f"or-quac-{target_name}.queries-manual.tsv")
        queries_raw = os.path.join("datasets/or-quac/queries/", f"or-quac-{target_name}.queries-raw.tsv")

        with open(target, "r") as f, open(queries_manual, "w") as g, open(queries_raw, "w") as i:
            for line in f:
                obj = json.loads(line)
                qid, query = obj['qid'], obj['rewrite']
                raw_query = obj["question"]

                g.write(f"{qid}\t{query}\n")
                i.write(f"{qid}\t{raw_query}\n")
                idx += 1

    print("Processing OR-QuAC qrel...")

    if not os.path.exists("datasets/or-quac/qrels"):
        os.makedirs("datasets/or-quac/qrels")

    with open("datasets/or-quac/raw/qrels.txt", "r") as f:
        qrels_dict = json.load(f)

    w_qrels = open("datasets/or-quac/qrels/or-quac.qrels.txt", "w")
    for qid, v in qrels_dict.items():
        for pid in v.keys():
            w_qrels.write(f"{qid} Q0 {pid} 1\n")
    w_qrels.close()

    print("Processing the collection of OR-QuAC...")
    if not os.path.exists("datasets/or-quac/jsonl"):
        os.makedirs("datasets/or-quac/jsonl")

    count = 0
    with open("datasets/or-quac/raw/all_blocks.txt", "r") as f, open("datasets/or-quac/jsonl/or-quac.jsonl", "w") as g:
        for line in tqdm(f):
            count+=1
            obj = json.loads(line)
            passage = obj['text'].replace('\n', ' ').replace('\t', ' ') # clear noises in text
            passage = obj["title"]+". "+passage
            pid = obj['id']
            g.write(json.dumps({"id": pid,"contents": passage}) + "\n")

    print(f"The number of passsages: {count}")
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    
    if args.dataset =="or-quac":
        ProcessORQuAC()
    elif args.dataset =="cast-19-20":
        ProcessCAsT()
    else:
        raise Exception