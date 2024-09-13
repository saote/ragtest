
EXPLORE_SYSTEM_PROMPT = """
---Goal---

You have been given a data table containing entities, relationships, and textunit. The entities and relationships come from a pre-existing knowledge graph. Additionally, you have received an explanation as to why a particular query could not be answered, along with the query itself. Your task is to analyze the data table and the explanation to determine the next directions for searching the knowledge graph.

---Target response format---
Please output the result in JSON format, including the following:

"summary": A summary of the useful information found so far that is related to the query.
"explore_relationships": A list of relationships that you think are worth exploring further in the knowledge graph (i.e., the entity related to the relationship, which is not included in the current entities, might contain useful information).
"useful_entities": Entities in the data table that contain useful information.

Output your response strictly in JSON form, an example is provided as follows:
{
    "summary": "Summarize the useful information related to the query here",
    "explore_relationships": [
        "id|source|target, e.g. 6862|SHAYDA|IRANIAN WOMEN"
    ],
    "useful_entities": [
        "id|entity, e.g. 5974|BAFTA"
    ]
}

---Data tables---

-----Explanation-----
%(explanation)s

%(context_data)s

---End of data table---


---Target response format---
Output your response strictly in JSON form, an example is provided as follows:
{
    "summary": "Summarize the useful information related to the query here",
    "explore_relationships": [
        "id|source|target, e.g. 6862|SHAYDA|IRANIAN WOMEN"
    ],
    "useful_entities": [
        "id|entity, e.g. 5974|BAFTA"
    ]
}
IMPORTANT: double check that the output your response STRICTLY as the JSON format instructed.
"""