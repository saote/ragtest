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


def read_and_chunk(input_file_dir, max_length=600):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    text = read_all_text_files(input_file_dir)

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
            current_chunk = current_chunk[-100:]
            current_length = 0

    # 确保最后一个块也被添加
    if current_chunk:
        chunks.append(tokenizer.convert_tokens_to_string(current_chunk))

    return chunks


def generate_response(question, documents, index: IndexFlatL2, model=None, k=10):
    # 转换问题为嵌入
    # question_embedding = model.encode([question])
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
    documents = read_and_chunk(input_file_dir)
    start_time = time.time()

    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # embeddings = model.encode(documents)
    embeddings = embed_document(documents)
    index = create_faiss_index(embeddings)

    end_time = time.time()
    duration = end_time - start_time
    print(f'indexing time: {format_time(duration)}')

    # read questions
    questions = read_questions(questions_file)
    # questions = [i + "\n用中文回答。" for i in questions]

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

