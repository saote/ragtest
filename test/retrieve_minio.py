from minio import Minio
from minio.error import S3Error
import markdown
from markdown.extensions.toc import TocExtension
from bs4 import BeautifulSoup
import os



# 创建MinIO客户端实例
client = Minio(
    "storage.minio.middleware.dev.motiong.net",
    access_key="uNGVqqdn0NWxYLkdpBjy",
    secret_key="Fk6JFGDtARSXx9mJown8r2dAk0OCe3M2yKTtQj1O",
    secure=True
)


def markdown_to_text(md_file_path):
    # 读取Markdown文件
    with open(md_file_path, 'r', encoding='utf-8') as file:
        md_content = file.read()

    # 转换Markdown到HTML
    html = markdown.markdown(md_content, extensions=[TocExtension()])

    # 使用BeautifulSoup解析HTML内容并获取纯文本
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()

    # 将纯文本写入新的.txt文件
    txt_file_path = md_file_path.replace('.md', '.txt')
    with open(txt_file_path, 'w', encoding='utf-8') as file:
        file.write(text)

    os.remove(md_file_path)
    print(f'Converted {md_file_path} to {txt_file_path}')


if __name__ == '__main__':
    bucket_name = "feishu-md"
    local_file_dir = "../input/knowledge_docs/FAE"

    try:
        objects = client.list_objects(bucket_name, recursive=False)
        for obj in objects:
            files_obj = client.list_objects(bucket_name, prefix=obj.object_name, recursive=False)
            for file in files_obj:
                file_name = file.object_name
                if file_name.endswith('labelled.md'):
                    local_file = local_file_dir + '/' + file_name.split('/')[-1]
                    client.fget_object(bucket_name, file_name, local_file)
                    print(f"下载文件：{obj.object_name} 到 {local_file}")
                    markdown_to_text(local_file)
    except S3Error as e:
        print('Error: ', e)

