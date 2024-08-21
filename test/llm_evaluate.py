import json
import csv
import numpy as np
import pandas as pd
from openai import OpenAI
from prompt import EVAL_SYSTEM_PROMPT
from prompt import get_eval_prompt
from utils import llm_generate_response, read_questions, create_empty_csv_file

def save_eval_data(question, eval_data, filename):
    new_data = [{
        'Question': question,
        'Comprehensiveness': eval_data['Comprehensiveness']['Decision'],
        'Diversity': eval_data['Diversity']['Decision'],
        'Empowerment': eval_data['Empowerment']['Decision'],
        'Directness': eval_data['Directness']['Decision'],
    }]

    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        # 将旧数据读取到列表中
        existing_data = list(reader)
        # 获取字段名（列名）
        fieldnames = reader.fieldnames

    # 将新行添加到数据列表中
    existing_data.extend(new_data)

    # 将更新后的数据写回到 CSV 文件中
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)


def evaluate(topic: str, questions_file: str):
    eval_file_name = f'../test_results/{topic}/evaluate.csv'
    rag_answer_file = f'../test_results/{topic}/RAG_result.csv'
    graph_rag_answer_file = f'../test_results/{topic}/GraphRAG_results.csv'

    # create empty csv file
    columns = ['Question', 'Comprehensiveness', 'Diversity', 'Empowerment', 'Directness']
    create_empty_csv_file(eval_file_name, columns)

    questions = read_questions(questions_file)

    df_rag = pd.read_csv(rag_answer_file)
    df_graph_rag = pd.read_csv(graph_rag_answer_file)

    for i, question in enumerate(questions):
        rag_answer = df_rag.loc[i, 'RAG Answer']
        graph_rag_answer = df_graph_rag.loc[i, 'Graph RAG answer']

        prompt = get_eval_prompt(question, rag_answer, graph_rag_answer)
        response = llm_generate_response(prompt)
        print(f'\n\n### {i} / {len(questions)} \nQ: {question}')
        print(response)
        response = json.loads(response)
        save_eval_data(question, response, eval_file_name)


if __name__ == '__main__':
    # input field
    input_dir = 'GraphRAG'
    # input_dir = 'stat_textbook'
    questions_file = './questions/graphrag_questions.txt'

    evaluate(input_dir, questions_file)


