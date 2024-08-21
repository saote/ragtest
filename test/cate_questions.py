from prompt import get_question_cat_prompt
from utils import llm_generate_response, read_questions
import json
import csv


def create_empty_csv_file(filename):
    columns = ['Question', 'Global_understanding', 'Reasoning_capabilities',
               'Deep_contextual_understanding', 'Question_type']
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def save_cat_data(cat_data, filename):
    cog_demands_data = cat_data['cognitive_demands']
    new_data = [{
        'Question': cat_data['question'],
        'Global_understanding': cog_demands_data['global_understanding']['required'],
        'Reasoning_capabilities': cog_demands_data['reasoning_capabilities']['required'],
        'Deep_contextual_understanding': cog_demands_data['deep_contextual_understanding']['required'],
        'Question_type': cat_data['question_type']['type'],
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


if __name__ == '__main__':
    question_file = './graphrag_questions.txt'
    cat_result_file = '../test_results/GraphRAG/questions_cat.csv'

    create_empty_csv_file(cat_result_file)

    questions = read_questions(question_file)
    for i, question in enumerate(questions):
        prompt = get_question_cat_prompt(question)
        answer = llm_generate_response(prompt)
        print(f'\n ### {i+1}/{len(questions)} \n' + answer)

        answer_data = json.loads(answer)
        save_cat_data(answer_data, cat_result_file)