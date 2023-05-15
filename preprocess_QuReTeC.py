import json
import argparse

def process(input_path, output_path):
    query = {}
    query_reader = open(input_path, 'r').readlines()
    new_query = open(output_path, "w")

    for idx, line in enumerate(query_reader):
        if idx==0:
            continue
        conversation_id, turn_id, id, qtext, original = line.split('\t')
        new_query.write("{}\t{}\n".format(id, qtext))

    new_query.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    process(args.input_path, args.output_path)

    #input_paths = ["datasets/cast-19-20/queries/5_QuReTeC_Q.tsv", "datasets/cast-19-20/queries/5_QuReTeC_QnA.tsv"]
    #output_paths = ["datasets/cast-19-20/queries/cast-19.queries-QuReTeC-Q.tsv","datasets/cast-19-20/queries/cast-20.queries-QuReTeC-QA.tsv"]