import csv
import json
import pandas as pd


def json_to_txt(input_file: str):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    output_file = input_file.replace('.json', '.txt')
    with open(output_file, 'w', encoding='utf-8') as text_file:
        for item in data:
            title = item.get('title', 'No Title Provided')
            author = item.get('author', 'Unknown author')
            source = item.get('source', 'No Source Provided')
            published_at = item.get('published_at', 'No Publish Date Provided')
            category = item.get('category', 'No Category Provided')
            url = item.get('url', 'No URL Provided')
            body = item.get('body', 'No Content Provided').replace('\n', ' ')

            # 将提取的数据写入到文本文件中
            text_file.write(f"Title: {title}\n")
            text_file.write(f"Author: {author}\n")
            text_file.write(f"Source: {source}\n")
            text_file.write(f"Published At: {published_at}\n")
            text_file.write(f"Category: {category}\n")
            # text_file.write(f"URL: {url}\n")
            text_file.write(f"Body: {body}\n")
            text_file.write("\n" + "=" * 80 + "\n\n")  # 添加分隔线和空行


def count_num_evidences(input_file):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    count = {}
    for item in data[:]:
        num_evidences = len(item.get('evidence_list', []))
        if num_evidences in count:
            count[num_evidences] += 1
        else:
            count[num_evidences] = 1
    return count


def extract_questions(question_file: str, num_question=100) -> list:
    with open(question_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    questions = []
    for item in data[:num_question]:
        question = item.get('query')
        questions.append(question)
    return questions


def extract_question_csv(question_file: str,  output_file: str) -> None:
    with open(question_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    with open(output_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data[0].keys())

        for item in data:
            writer.writerow([
                item['query'],
                item['answer'].replace('Insufficient information.', 'Insufficient Information'),
                item['question_type'],
                len(item['evidence_list'])
            ])


if __name__ == '__main__':
    DATA_FILE = './corpus.json'
    QUESTION_FILE = './MultiHopRAG.json'
    OUTPUT_CSV_FILE = '../test_results/multi_hop/questions.csv'
    # json_to_txt(DATA_FILE)
    # count = count_num_evidences(QUESTION_FILE)

    extract_question_csv(QUESTION_FILE, OUTPUT_CSV_FILE)

