import subprocess
import csv
import json
import pandas as pd
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from utils import read_questions, create_empty_csv_file, save_answer_data
import graphrag
import time
import re


def run_command(command):
    """Run a single terminal command."""
    try:
        # print(command[-1])
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        output = output.split("SUCCESS: Local Search Response:", 1)[-1].strip()

        return output

    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error:\n{e.stderr}")
        return None


def batch_commands(commands):
    results = {}
    for cmd in commands:
        result = run_command(cmd)
        results[cmd[-1]] = result
    return results


def save_dict_to_file(data_dict, filename):
    """Save the dictionary to a file in JSON format."""
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_dict, file, ensure_ascii=False, indent=4)


def clean_response(response: str) -> str:
    pattern = r'\[Data:.*?\]'
    return re.sub(pattern, '', response)


def create_clean_response(filename: str) -> None:
    df = pd.read_csv(filename)
    df['Graph RAG clean response'] = df['Graph RAG answer'].apply(clean_response)
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    # input field
    topic = 'multi_hop'
    question_file = './questions/multi_hop_questions.txt'
    # indexing_data_dir = '../output/20240717-143037_stat/artifacts'
    question_start_index = 0

    result_file = f'../test_results/{topic}/GraphRAG_results.csv'

    # read questions
    questions = read_questions(question_file)

    # generate response
    cmd = [
        "python3", "-m", "graphrag.query",
        "--root", "../",
        # "--data", indexing_data_dir,
        "--method", "local",
    ]

    columns = ['Question', 'Graph RAG answer']

    # create empty answer file
    if question_start_index == 0:
        create_empty_csv_file(result_file, columns)

    start_time = time.time()
    # query each question
    for i, query in enumerate(questions[question_start_index:]):
        query_cmd = cmd + [query]
        print(f"\n\n{i}/{len(questions[question_start_index:])}\n### Q: " + query)

        answer = run_command(query_cmd)

        print('### A:' + answer + '\n')
        save_answer_data(data=[query, answer], column_names=columns, filename=result_file)

    # calculate the time
    end_time = time.time()
    duration = end_time - start_time
    print(f'execution time: {int(duration // 60)} min {int(duration % 60)} sec')
    avg_time = duration / len(questions[question_start_index:])
    print(f'average time per question: {int(avg_time // 60)} min {int(avg_time % 60)} sec')

    create_clean_response(result_file)


