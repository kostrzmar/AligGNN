
from graph_builder.processor.coreNLP_processor import coreNLPProcessor
from graph_builder.processor.amr_processor import AMRProcessor
from graph_builder.processor.stanza_processor import StanzaProcessor
from graph_builder.processor.spacy_processor import SpacyProcessor
import logging

class DataConverterProcessor():
    def __init__(self, 
                params=None,
                ) -> None:
        self.params = params
        self.processor_name = self.params["graph.builder.processor"]
        self.builder_names = []    
        self.data_set = None
        self.coreNLP = None
        self.amr = None
        self.stanza = None
        self.spacy = None

    def registerBuilderName(self, builder_name):
        self.builder_names.append(builder_name)

        
    def initializeProcessors(self, data_set, builders_name):
        
        self.data_set = data_set
        for builder_name in builders_name:
            self.registerBuilderName(builder_name)
        if not self.amr and 'Amr' in self.builder_names:
            logging.info(f'Registering processor [Amr]')
            self.amr = AMRProcessor(params=self.params, data_set=self.data_set)
            
        if self.processor_name == "coreNLP":   
            if  not self.coreNLP and any(item in self.builder_names for item in ['Sequence','Dependency', 'Ie', 'POS', 'Constituency', "Amr"]):
                logging.info(f'Registering processor [CoreNLP]')
                self.coreNLP = coreNLPProcessor(params=self.params, url='http://localhost', port=9000, data_set=self.data_set)
                self.coreNLP.registerAnnotatorsForBuilders(self.builder_names)
                for builder_name in builders_name:
                    self.coreNLP.registerAnnotatorsForBuilders(builder_name)
            elif not self.coreNLP:
                logging.error(f'Builder name [{builders_name}] not supported for [{self.processor_name}]')
        elif self.processor_name == "stanza":   
            if not self.stanza and any(item in self.builder_names for item in ['POS', 'Dependency', 'Constituency']):
                logging.info(f'Registering processor [Stanza]')
                self.stanza = StanzaProcessor(params=self.params, data_set=self.data_set)
                self.stanza.registerAnnotatorsForBuilders(self.builder_names)
            elif not self.stanza:
                logging.error(f'Builder name [{builders_name}] not supported for [{self.processor_name}]')
        elif self.processor_name == "spacy":   
            if not self.spacy and any(item in self.builder_names for item in ['POS', 'Dependency']):
                logging.info(f'Registering processor [Spacy]')
                self.stanza = SpacyProcessor(params=self.params, data_set=self.data_set)
                self.stanza.registerAnnotatorsForBuilders(self.builder_names)
            elif not self.stanza:
                logging.error(f'Builder name [{builders_name}] not supported for [{self.processor_name}]')                
                
              
    def preprocess_graph(self, dataset, sentence, sentence_mask):
        if self.amr:
           self.amr.preprocess_graph(dataset,sentence, sentence_mask)
        if self.coreNLP:
           self.coreNLP.preprocess_graph(dataset,sentence, sentence_mask) 
        if self.stanza:
           self.stanza.preprocess_graph(dataset,sentence, sentence_mask)  
        if self.spacy:
           self.spacy.preprocess_graph(dataset,sentence, sentence_mask)   
        
    def getProcessor(self, builder_name):
        if builder_name == 'Amr':
            return self.amr
        else:
            if self.processor_name == "coreNLP":
                return self.coreNLP
            elif self.processor_name == "stanza":   
                return self.stanza
            elif self.processor_name == "spacy":   
                return self.spacy
            else:
                logging.error(f'Processor not supported for [{self.processor_name}]')
                return None
