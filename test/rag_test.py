from sentence_transformers import SentenceTransformer
import faiss
from faiss import IndexFlatL2
import numpy as np
from openai import OpenAI
from transformers import BertTokenizer
import pandas as pd
from utils import read_questions, llm_generate_response, embed_document, create_faiss_index
import time
import os
from prompt import (RAG_SYS_PROMPT)


def get_entity_info_multi_hop(text: str, entity_name: str)->str:
    entity_chunks = text.split('\n')
    entity_content = ''
    for lines in entity_chunks:
        if lines.startswith(entity_name + ':'):
            entity_content = lines[len(entity_name) + 2:]
    return entity_content


def format_time(seconds):
    return f'{int(seconds // 60)} min {int(seconds % 60)} sec'

def read_all_text_files(input_dir:str) -> str:
    output = ''
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                output += file.read()
                output += '\n'

    return output


def read_all_text_files_to_list(input_dir:str) -> list[str]:
    output = []
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                output.append(file.read())
    return output


def chunk_txt(text:str, max_length:int=600, overlap:int=100) -> list[str]:
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    # 使用 tokenizer 进行分词，但保持分块不超过 max_length
    tokens = tokenizer.tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        current_chunk.append(token)
        current_length += 1
        # 当累积到一定数量的 token 或达到最大长度限制时，合并为一个块
        if current_length >= max_length:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
            current_chunk = current_chunk[-overlap:]
            current_length = 0

    # 确保最后一个块也被添加
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

    return chunks


def read_and_chunk_multi_hop(input_file_dir:str, max_length:int=700) -> list[str]:
    text_list = read_all_text_files_to_list(input_file_dir)

    chunks = []
    for text in text_list:
        title = get_entity_info_multi_hop(text, 'Title')
        source = get_entity_info_multi_hop(text, 'Source')
        published_at = get_entity_info_multi_hop(text, 'Published At')
        body = get_entity_info_multi_hop(text, 'Body')
        current_chunks = chunk_txt(text=body, max_length=max_length)
        for chunk in current_chunks:
            chunk += ('\nTitle: ' + title)
            chunk += ('\nSource: ' + source)
            chunk += ('\nPublish Time: ' + published_at)
            chunks.append(chunk)

    return chunks


def generate_response(question, documents, index: IndexFlatL2, k=15):
    # 转换问题为嵌入
    question_embedding = embed_document([question])
    D, I = index.search(np.array(question_embedding).astype(np.float32), k=k)  # 检索最相关的文档
    retrieved_document = [documents[i] + '\n' for i in I[0]]

    # 创建提示文本
    system_prompt = RAG_SYS_PROMPT % {
        'context_data': retrieved_document
    }
    prompt = f'Question: {question}'

    response = llm_generate_response(prompt, system_prompt)
    return response


def save_answers(answers, questions, filename):
    df = pd.DataFrame({
        'Question': questions,
        'RAG Answer': answers
    })
    df.to_csv(filename, index=False, encoding='utf-8')


def rag_test(questions_file: str, topic:str):
    input_file_dir = '../input'
    save_file = f'../test_results/{topic}/RAG_result.csv'
    question_start_index = 0

    # indexing
    # text = read_all_text_files(input_file_dir)
    # documents = chunk_txt(text)
    documents = read_and_chunk_multi_hop(input_file_dir)
    print('finish chunking...')

    start_time = time.time()
    embeddings = embed_document(documents)
    index = create_faiss_index(embeddings)
    end_time = time.time()

    duration = end_time - start_time
    print(f'indexing time: {format_time(duration)}')

    # read questions
    questions = read_questions(questions_file)

    # generate answers
    start_time = time.time()
    answers = []
    for i, question in enumerate(questions[question_start_index:]):
        print(f"\n\n{i+1}/{len(questions[question_start_index:])}\n### Q: " + question)
        answer = generate_response(question, documents, index)
        print(answer)
        answers.append(answer)

    save_answers(answers, questions, save_file)

    # calculate the time
    end_time = time.time()
    duration = end_time - start_time
    print(f'execution time: {format_time(duration)}')
    avg_time = duration / len(questions[question_start_index:])
    print(f'average time per question: {format_time(avg_time)}')


def no_context_response(questions_file: str, topic: str):
    save_file = f'../test_results/{topic}/no_context_result.csv'
    question_start_index = 0

    # read questions
    questions = read_questions(questions_file)

    # generate answers
    start_time = time.time()
    answers = []
    for i, question in enumerate(questions[question_start_index:]):
        print(f"\n\n{i + 1}/{len(questions[question_start_index:])}\n### Q: " + question)

        system_prompt = RAG_SYS_PROMPT % {
            'context_data': 'None'
        }
        prompt = f'Question: {question}'
        answer = llm_generate_response(prompt, system_prompt)

        print(answer)
        answers.append(answer)

    save_answers(answers, questions, save_file)

    # calculate the time
    end_time = time.time()
    duration = end_time - start_time
    print(f'execution time: {format_time(duration)}')
    avg_time = duration / len(questions[question_start_index:])
    print(f'average time per question: {format_time(avg_time)}')


if __name__ == '__main__':
    questions_file = 'questions/multi_hop_questions.txt'
    # topic = 'stat_textbook'
    topic = 'multi_hop'

    rag_test(questions_file, topic)
    # no_context_response(questions_file, topic)


