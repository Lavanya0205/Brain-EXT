import networkx as nx
class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()

    def add_concepts(self, concepts):
        for concept in concepts:
            self.graph.add_node(concept)

    def connect(self, c1, c2):
        if self.graph.has_edge(c1, c2):
          self.graph[c1][c2]["weight"] += 1
        else:
         self.graph.add_edge(c1, c2, weight=1)

knowledge_graph = KnowledgeGraph()