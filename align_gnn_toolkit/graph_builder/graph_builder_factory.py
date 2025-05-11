
from graph_builder.impl.graph_builder_sequence import GraphBuilderSequence
from graph_builder.impl.graph_builder_master import GraphBuilderSequenceMaster
from graph_builder.impl.graph_builder_dependency import GraphBuilderSequenceDependency
from graph_builder.impl.graph_builder_ie import GraphBuilderSequenceIE
from graph_builder.impl.graph_builder_pos import GraphBuilderSequencePOS
from graph_builder.impl.graph_builder_constituency import GraphBuilderSequenceConstituency
from graph_builder.impl.graph_builder_amr import GraphBuilderAMR
from graph_builder.impl.graph_builder_document_hierarchy import GraphBuilderDocumentHierarchy
from graph_builder import GraphBuilder
from utils import config_const
import logging

class GraphBuilderFactory():
    
    @staticmethod
    def getBuilder(params=None):
        try:
            graph_builder = GraphBuilder(params=params)
            if config_const.CONF_GRAPH_ONLY_ARM in params and params[config_const.CONF_GRAPH_ONLY_ARM] and len(params[config_const.CONF_GRAPH_BUILDER_NAME])==1:
                logging.info(f'Skipping sequence builder since only AMR')
            else:
                graph_builder.registerBuilder(GraphBuilderSequence(params=params))
            builders = params[config_const.CONF_GRAPH_BUILDER_NAME]
            for builder in builders:
                if builder == "dependency":
                    graph_builder.registerBuilder(GraphBuilderSequenceDependency(params=params)) 
                elif builder == "master":
                    graph_builder.registerBuilder(GraphBuilderSequenceMaster(params=params)) 
                elif builder == "ie":
                    graph_builder.registerBuilder(GraphBuilderSequenceIE(params=params))
                elif builder == "pos":
                    graph_builder.registerBuilder(GraphBuilderSequencePOS(params=params))
                elif builder == "constituency":
                    graph_builder.registerBuilder(GraphBuilderSequenceConstituency(params=params))
                elif builder == "amr":
                    graph_builder.registerBuilder(GraphBuilderAMR(params=params))
                elif builder == "documentHierarchy":
                    graph_builder.registerBuilder(GraphBuilderDocumentHierarchy(params=params))
            return graph_builder
        except AssertionError as e:
            logging.error(f'Error during initialization of graph builders [{e}]')