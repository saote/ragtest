from graphrag.model import Entity
from graphrag.query.llm.base import BaseLLM
from explore_graph.search_graph import GraphSearcher
from graphrag.query.structured_search.base import BaseSearch, SearchResult
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.system_prompt import (
    EXPLORE_MULTIHOP_LOCAL_SEARCH_SYS_PROMPT,
)
from explore_graph.prompt import (
    EXPLORE_SYSTEM_PROMPT,
)

from explore_graph.base import QueryResult, ExploreResult

import json
import time
from typing import Any


DEFAULT_LLM_PARAMS = {
    "max_tokens": 1500,
    "temperature": 0.0,
}

class LocalExplore():
    def __init__(
            self,
            search_engine: BaseSearch,
            llm: BaseLLM,
            context_builder: LocalSearchMixedContext,
            query_system_prompt: str = EXPLORE_MULTIHOP_LOCAL_SEARCH_SYS_PROMPT,
            explore_system_prompt:str = EXPLORE_SYSTEM_PROMPT,
            llm_params: dict[str, Any] = DEFAULT_LLM_PARAMS,
            explore_context_params: dict | None = None,
            ) -> None:
        self.llm = llm
        self.search_engine = search_engine
        self.explore_system_prompt = explore_system_prompt
        self.context_builder = context_builder
        self.explore_context_params = explore_context_params
        self.llm_params = llm_params
        self.query_system_prompt = query_system_prompt

        self.graph_searcher = GraphSearcher(entities=context_builder.entities, relationships=context_builder.relationships)

    async def explore(
            self,
            query: str,
            max_iter: int = 5,
            ) -> QueryResult:
        start_time = time.time()

        query_result:SearchResult = await self.search_engine.asearch(query)
        try:
            query_response = json.loads(query_result.response)
        except ValueError:
            raise ValueError(f'query response "{query_response}" can not be parsed properly')

        try:
            sufficient_info = query_response['sufficient_info'].lower()
            answer = query_response['answer']
            explanation = query_response['explanation']

            if sufficient_info == 'yes':
                
                return QueryResult(
                    find_answer = True,
                    response=answer,
                    explanation = explanation,
                    context_data=query_result.context_data,
                    context_text=query_result.context_text,
                    completion_time=time.time() - start_time,
                    num_iter=0,
                    prompt_tokens=query_result.prompt_tokens,
                )
            
            elif sufficient_info != 'no':
                raise ValueError(f'sufficient_info: {sufficient_info} is not allowed')

        except KeyError:
            raise KeyError(f'query response {query_response} can not be parsed properly.')

        num_iter = 0
        
        current_context_text = query_result.context_text
        current_explanation = explanation
        current_answer = answer
        current_context_data = query_result.context_data
        # this set keeps the history of explored entities set. When a new explore entities set is explored before, early stop.
        explored_entities_set = set()
        while num_iter < max_iter:
            num_iter += 1
            explore_response:ExploreResult = await self._explore_graph(explanation= current_explanation, context_text=current_context_text, query=query)
            explore_entities = explore_response.selected_entities
            explore_entities_title = [e.title for e in explore_entities]
            current_summary = explore_response.summary

            print(f'iteration: {num_iter}')
            print(f'answer: {current_answer}')
            print(f'entities to explore: {[entity.title for entity in explore_entities]}')
            print(f'summary: {current_summary}\n')
            ### if there is no more entity to explore, or explore entities have already been explored, return the previous response
            if len(explore_entities) == 0 or frozenset(explore_entities_title) in explored_entities_set:
                return QueryResult(
                    find_answer = False,
                    response=current_answer,
                    explanation = current_explanation,
                    context_data=current_context_text,
                    context_text=current_context_data,
                    completion_time=time.time() - start_time,
                    num_iter=num_iter
                )
            explored_entities_set.add(frozenset(explore_entities_title))
            query_response:QueryResult = await self._query_llm_with_entities(
                explore_entities=explore_entities,
                summary = current_summary,
                query=query,
            )

            query_response.completion_time = time.time() - start_time
            query_response.num_iter = num_iter
            if query_response.find_answer:
                return query_response
            
            current_context_text = query_response.context_text
            current_explanation = query_response.explanation
            current_context_data = query_response.context_data

        return query_response


    async def _explore_graph(
            self,
            explanation: str,
            context_text: str,
            query: str,
    ) -> ExploreResult:
        """ return a list of entities that LLM thinks worth exploring on the next steps"""
        system_prompt = self.explore_system_prompt % {
            'explanation': explanation,
            'context_data': context_text
        }
        search_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        llm_response = await self.llm.agenerate(
            messages=search_messages,
            streaming=True,
            **self.llm_params,
        )
        try:
            llm_response = json.loads(llm_response)
        except ValueError:
            return ExploreResult(
                summary=llm_response,
                selected_entities=[],
            )
        
        try:
            useful_entities = llm_response['useful_entities']
            explore_relationships = llm_response['explore_relationships']
            summary = llm_response['summary']
        except KeyError:
            raise KeyError(f'llm response {llm_response} can not be parsed properly')
        
        # if there is no relationships to explore, there will be no additional information gained. return an empty list
        if len(explore_relationships) == 0:
            return ExploreResult()
        
        # extract entities and relationships from ids
        selected_entities = self._extract_selected_entities(
            entities=useful_entities,
            explore_relationships=explore_relationships,            
            )

        return ExploreResult(
            summary=summary,
            selected_entities = selected_entities,
        )
        

    async def _query_llm_with_entities(
            self,
            explore_entities: list[Entity],
            query: str,
            summary: str = 'None'
    )->QueryResult:
        """ Query LLM with  additional entities that must be included in the context data"""
        llm_result = await self.search_engine.asearch(
            query=query,
            summary = summary,
            include_entity_names = [entity.title for entity in explore_entities],
        )

        llm_response = llm_result.response
        try:
            llm_response = json.loads(llm_response)
            sufficient_info = llm_response['sufficient_info'].lower()
            answer = llm_response['answer']
            explanation = llm_response['explanation']
            if sufficient_info == 'yes':
                return QueryResult(
                    find_answer = True,
                    response=answer,
                    explanation = explanation,
                    context_data=llm_result.context_data,
                    context_text=llm_result.context_text,
                )
            
            elif sufficient_info == 'no':
                return QueryResult(
                    find_answer = False,
                    response=answer,
                    explanation = explanation,
                    context_data=llm_result.context_data,
                    context_text=llm_result.context_text,
                )

            else:
                raise ValueError(f'sufficient_info: {sufficient_info} is not allowed')

        except KeyError:
            raise KeyError(f'llm response {llm_response} can not be parsed properly.')


    def _extract_selected_entities(
            self, 
            entities: list[str],
            explore_relationships: list[str] = [],
    )->list[Entity]:
        """ Extract related entities from useful entities and relationships to be explored"""
        all_entities = self.graph_searcher.entities
        selected_entities_id = set()
        for entity in entities:
            id, title = entity.split('|')
            if id in all_entities.keys() and all_entities[id].title == title:
                selected_entities_id.add(id)

        relationship_ids = []
        for relationship in explore_relationships:
            relationship_ids.append(relationship.split('|')[0])
        related_entities = self.graph_searcher.get_entities_of_relationships(relationship_ids)

        selected_entities_id.update(related_entities)

        selected_entities = []
        for id in selected_entities_id:
            selected_entities.append(all_entities[id])

        return list(selected_entities)
        




    
        
