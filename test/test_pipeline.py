from llm_evaluate import evaluate
from rag_test import rag_test
# from graph_rag_test import

if __name__ == '__main__':
    question_file = ''
    topic = 'stat_textbook'

    # RAG llm response
    rag_test(questions_file=question_file, topic=topic)

    # Graph RAG test

    # evaluate


