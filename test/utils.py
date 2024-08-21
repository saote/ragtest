import openai
from openai import OpenAI
import csv
import pandas as pd
import numpy as np
from faiss import IndexFlatL2
import faiss
import math

def llm_generate_response(user_prompt, sys_prompt=None)->str:
    model_name = 'qwen2-instruct'
    base_url = 'http://10.4.32.1:9997/v1'
    # base_url = 'http://10.4.32.1:29997/v1'
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
    base_url = 'http://10.4.32.2:9997/v1'
    model = "bge-m3"

    client = OpenAI(api_key='key', base_url=base_url)
    response = client.embeddings.create(
        input=document,
        model=model
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


if __name__ == '__main__':
    # input_file = '../input/场景汇总.xlsx'
    # csv_file = excel_to_csv(input_file)
    # csv_to_txt_files(csv_file)
    query = 'Is apple a company or type of food?'
    query_tuple = (3957, 24149, 264, 2883, 477, 955, 315, 3691, 30)
    input1 = 'APPLE:Apple, a multinational technology company based in Cupertino, is renowned for its innovative products and services, including consumer electronics, computer software, and online services. The company designs, manufactures, and sells a wide range of devices such as iPhones, iPads, MacBooks, Apple Watches, AirPods, and accessories like the Magic Trackpad, Magic Mouse, and Magic Keyboard. Apple\'s product line also includes the MacBook Pro and MacBook Air laptops, equipped with its own M-series chips, and the Mac Mini desktop computer. The company is the manufacturer of the MacBook Air and the Apple Watch Ultra 2, known for its innovative electronic devices and technology.\n\nApple is a leader in the app store market, operating the iOS App Store, which features separate lists of top apps and games for iPhone and iPad, both free and paid, and offers Apple Arcade subscription gaming. The company has been involved in legal disputes over app store fees and policies, notably with Epic Games. Apple\'s App Store is a platform for iOS apps, competing with Google\'s Android operating system and Google Play Store.\n\nApple\'s technology prowess extends to its AirPods Pro, considered the best earbuds for iPhone users with excellent noise canceling and microphones. The AirPods Pro (USB-C) and the second-generation AirPods are currently on sale. Apple also produces the AirPods Max and has a spatial audio feature that allows for a more immersive experience when listening to or watching compatible content.\n\nApple\'s smartwatch, the Apple Watch, offers comprehensive health and fitness tracking, iPhone notifications, crash and fall detection, noise monitoring, and Emergency SOS. The company is reportedly working on sleep apnea detection for the Apple Watch, which would require a lengthy FDA clearance process. The Apple Watch Series 6 is at the center of a patent dispute with Masimo, and Apple has faced import bans and cease-and-desist orders due to disputes over patent infringement related to the Apple Watch\'s EKG features.\n\nApple\'s relationship with Google is significant, as the company received about $18 billion from Google in 2021 as part of a deal that made Google the default search engine on Apple devices. Apple has not switched to a Google competitor or allowed users to choose their browser when setting up their iPhones. However, Apple does offer users choices about various features and services, including search engines.\n\nApple is known for its refined and polished products and is poised to enter the AI market with a focus on practical applications of users'
    input2 = 'SHE:She is a person who experienced severe anxiety and nightmares due to a concerning student situation, and advocated for support from people power and reporting to the Title IX office'
    embeddings = embed_document([input1, input2])
    index = create_faiss_index(embeddings)

    question_embedding = embed_document(query)
    D, I = index.search(np.array(question_embedding).astype(np.float32), k=2)
    # print(embeddings)
    print(question_embedding)
    print(D)
    print(I)




