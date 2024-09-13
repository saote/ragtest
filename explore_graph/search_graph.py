from graphrag.model import Covariate, Entity, Relationship

class GraphSearcher():
    def __init__(
            self,
            entities: dict,
            relationships: dict | None = None,
            ) -> None:
        self.entities = {entity.short_id: entity for entity in entities.values()}
        self.relationships = {relationship.short_id: relationship for relationship in relationships.values()}

    def get_entities_of_relationship(self, relationship_id: str) -> list[str]:
        """return source and target entity of a relationship given relationship id"""
        if relationship_id not in self.relationships.keys():
            return []
        relationship = self.relationships[relationship_id]
        source_entity_id = [entity.short_id for entity in self.entities.values()
                         if entity.title == relationship.source]
        target_entity_id = [entity.short_id for entity in self.entities.values()
                         if entity.title == relationship.target]
        return source_entity_id + target_entity_id
    

    def get_entities_of_relationships(self, relationship_ids: list[str]) -> set[str]:
        """return source and target entities id of a list of relationships given relationship ids"""
        if len(relationship_ids) == 0:
            return []
        related_entities = set()
        for relationship_id in relationship_ids:
            current_related_entities = self.get_entities_of_relationship(relationship_id)
            related_entities.update(current_related_entities)
        return related_entities
    

    def get_entity_neighbors(self, entity_id: str) -> list[str]:
        """return all neighbor entities of one entity"""
        if entity_id not in self.entities.keys():
            return []
        
        entity:Entity = self.entities[entity_id]
        entity_name = entity.title
        source_neighbors = [relationship.source for relationship in self.relationships
                            if relationship.target == entity_name]
        
        target_neighbors = [relationship.target for relationship in self.relationships
                            if relationship.source == entity_name]
        
        return source_neighbors + target_neighbors


        



