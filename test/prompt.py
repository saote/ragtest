EVAL_PROMPT = """
Question: %(question)s

Please evaluate the answers provided below based on the specified metrics. For each metric, determine which answer (0 or 1) is superior, or state if it's a tie. Please provide a concise justification for your decision for each metric.

Answer 0: %(answer0)s
Answer 1: %(answer1)s

Evaluation Criteria:
1. **Comprehensiveness**: Measures how thoroughly each answer covers all aspects and details of the question. 
   - Decision (0/1/Tie): 
   - Justification: 

2. **Diversity**: Assesses the variety and richness of different perspectives and insights each answer offers on the question.
   - Decision (0/1/Tie): 
   - Justification: 

3. **Empowerment**: Evaluates how effectively each answer aids the reader in understanding and making informed judgments about the topic.
   - Decision (0/1/Tie): 
   - Justification: 

4. **Directness**: Examines how specifically and clearly each answer addresses the question directly.
   - Decision (0/1/Tie):
   - Justification: 

Please provide your evaluations strictly as below in JSON format. Double check to make sure it is correct.
{
     "Comprehensiveness": {
        "Decision": "Your Decision Here",
        "Justification": "Your Justification Here"
    },
    "Diversity": {
        "Decision": "Your Decision Here",
        "Justification": "Your Justification Here"
    },
    "Empowerment": {
        "Decision": "Your Decision Here",
        "Justification": "Your Justification Here"
    },
    "Directness": {
        "Decision": "Your Decision Here",
        "Justification": "Your Justification Here"
    }
}
"""

EVAL_SYSTEM_PROMPT = """Given the question and two responses provided, your task is to evaluate the responses based on 
the following criteria: Comprehensiveness, Diversity, Empowerment, and Directness. For each criterion, choose whether 
Response 0 or Response 1 is better, or if they are equal (Tie). Provide a brief justification for each decision."""


QUESTION_CAT_PROMPT = """Evaluate the following question to determine if it requires global 
understanding, reasoning capabilities, and deep contextual understanding. Classify the type of question as either 
factual, explanatory, or reasoning. Provide your response in strict JSON format.

Question: %(question)s

Instructions:

Assess Cognitive Demands:

Global Understanding: Does the question require a broad holistic understanding of the topic? 
Reasoning Capabilities: Is complex multiple-step logical deduction necessary to respond effectively?
Deep Contextual Understanding: Are detailed, context-specific insights essential for answering?
Classify Question Type:

Limit your classification to factual, explanatory, reasoning, or others.

Do not mark the cognitive demand with true unless you believe there is a strong inclination that such demand is required.
Format Response:

Provide your analysis and classification in JSON format, clearly labeling each part of the assessment.
Example JSON Output:
{
  "question": 'sample question',
  "cognitive_demands": {
    "global_understanding": {
      "required": true/false,
      "reason": "your reason here"
    },
    "reasoning_capabilities": {
      "required": true/false,
      "reason": "your reason here"
    },
    "deep_contextual_understanding": {
      "required": true/false,
      "reason": "your reason here"
    }
  },
  "question_type": {
    "type": "explanatory/factual/reasoning",
    "reason": "your reason here"
  }
}
"""

RAG_SYS_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response of the target length and format that responds to the user's question based on the context.

ONLY use the context data as your knowledge base and do NOT include any general knowledge you already known.
If the provided context data is insufficient to answer the question, respond 'Insufficient information'

---Target response length and format---

A word or entity. Answer directly WITHOUT explanation.

---Context Data---

%(context_data)s

"""


def get_eval_prompt(
        question: str,
        answer0: str,
        answer1: str
):
    return EVAL_PROMPT % {
        'question': question,
        'answer0': answer0,
        'answer1': answer1
    }


def get_question_cat_prompt(question:str):
    return QUESTION_CAT_PROMPT % {
        'question': question
    }


if __name__ == '__main__':
    print(get_eval_prompt('why?', '000', '1111'))