import openai
from openai import OpenAI
import csv
import pandas as pd
import numpy as np
from faiss import IndexFlatL2
import faiss


LLM_MODEL_NAME = 'llama-3.1-instruct'
LLM_BASE_URL = 'http://10.4.32.1:9997/v1'
EMBED_MODEL_NAME = 'bge-m3'
EMBED_BASE_URL = 'http://10.4.32.1:9997/v1'

def llm_generate_response(user_prompt:str, sys_prompt:str=None)->str:
    model_name = LLM_MODEL_NAME
    base_url = LLM_BASE_URL
    client = OpenAI(api_key='not empty', base_url=base_url)

    # 请求 OpenAI 完成
    if sys_prompt:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

    return response.choices[0].message.content


def read_questions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = file.read().split('\n')
    return questions


def create_empty_csv_file(filename, columns):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def excel_to_csv(input_file_path:str) -> str:
    data = pd.read_excel(input_file_path)
    output_file_path = input_file_path.replace('.xlsx', '.csv')
    data.to_csv(output_file_path)
    return output_file_path


# convert every row of csv file into a txt chunks
def csv_to_txt_files(input_file_path: str) -> None:
    data = pd.read_csv(input_file_path)
    headers = data.columns.to_list()

    for index, row in data.iterrows():
        content = '\n'.join(f"'{header}': '{row[header]}'" for header in headers)

        with open(f'../input/row_{index}.txt', 'w', encoding='utf-8') as file:
            file.write(content)


def embed_document(document: list | tuple | str | int):
    client = OpenAI(api_key='key', base_url=EMBED_BASE_URL)

    response = client.embeddings.create(
        input=document,
        model=EMBED_MODEL_NAME
    )

    embedding_data = response.data
    return np.array([np.array(i.embedding) for i in embedding_data])


def create_faiss_index(embeddings) -> IndexFlatL2:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype(np.float32))
    return index


def l2_distance(p, q):
    return sum((px - qx) ** 2 for px, qx in zip(p, q))


def save_answer_data(data: list, column_names: list[str], filename:str):
    new_data = [{
        column_names[i]: data[i] for i in range(len(column_names))
    }]
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        existing_data = list(reader)
        fieldnames = reader.fieldnames

    existing_data.extend(new_data)

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)


if __name__ == '__main__':
    pass