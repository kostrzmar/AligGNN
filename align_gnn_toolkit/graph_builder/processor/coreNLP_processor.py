from graph_builder.processor.processor_abstract import ProcessorAbstract
from data_set.data_set_abstract import DatasetAbstract
from data_set.data_set_processor import DataSetProcessor
from stanfordcorenlp import StanfordCoreNLP
from collections import deque
import json

class coreNLPProcessor(ProcessorAbstract):
    def __init__(self, 
            params=None,
            url=None,
            port=None,
            data_set=None
            ) -> None:
        super(coreNLPProcessor, self).__init__(params, url, port, data_set)

        self.coreNLP = StanfordCoreNLP(url, port=port, timeout=300000, lang=self.lang)
        self.coreNLPAnnotators = ["ssplit","tokenize"]
        self.annotators = {
            "Constituency":["parse"], 
            "Dependency":["pos","depparse"],
            "IE":["openie"], 
            "POS":["pos"]
        }

    def getProcessorName(self):
        return "coreNLP"
    
    def registerAnnotatorsForBuilders(self, builder_names):
        for builder_name in builder_names:
            if builder_name in self.annotators:
                for annotator in self.annotators[builder_name]:
                    if annotator not in self.coreNLPAnnotators:
                        self.coreNLPAnnotators.append(annotator)
        
    def getAnnotators(self):
        return ",".join(self.coreNLPAnnotators)
    
    def getProcessorArguments(self):
        return  {
                'annotators': self.getAnnotators(),
                'pipelineLanguage':self.lang,
                'tokenize.language': self.lang,
                "tokenize.whitespace": False,
                'ssplit.isOneSentence': True,
                'outputFormat': 'json'
            }
    
    """    
    def preprocess_graph(self, sentence, sentence_mask):
        self.sentence_dict,self.sentence_out_mapping = self.parseSentence(sentence, sentence_mask)
    """
    def preprocess_graph(self, dataset,item, item_type):
        self.sentence_dict  = self.get_annotations(item[dataset.get_token_name(item_type,DatasetAbstract.SENTENCE_TOKEN_NAME)])

    def get_annotations(self, sentence_as_text):
        sentence_as_hash = None
        core_nlp_dict = None
        processor_args = self.getProcessorArguments()
        if self.isCacheEnabled():
            sentence_as_hash = self.getHashFromSentence(sentence_as_text, "_".join(str(i) for i in self.coreNLPAnnotators))
            if self.isInCache(sentence_as_hash):
                core_nlp_dict = self.getFromCache(sentence_as_hash)
        
        if not core_nlp_dict:
            core_nlp_json = self.coreNLP.annotate(sentence_as_text.strip(), properties=processor_args)
            core_nlp_dict = json.loads(core_nlp_json)
            if self.isCacheEnabled():
                self.setToCache(sentence_as_hash, core_nlp_dict) 
        return core_nlp_dict


    def parseSentence(self, sentence, sentence_mask):        
        core_nlp_out_mapping = None
        core_nlp_dict = None
        sentence_as_string = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_sentence_as_text(sentence)
        core_nlp_dict = self.get_annotations(sentence_as_string) 
            
        if  len(core_nlp_dict['sentences'][0]['tokens']) != len(sentence_mask):
            core_nlp_out_mapping = {}
            token_to_emb=[]
            input_items = []
            for item in sentence:
                input_items.append({"token_ids":item})
                
            for s_id in range(len(core_nlp_dict["sentences"])):
                for tokens in core_nlp_dict["sentences"][s_id]["tokens"]:
                    str_tokens = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_tokenizer(tokens["word"])
                    #str_tokens_ids = [self.data_set.vocab.get_stoi()[x] for x in str_tokens]
                    str_tokens_ids = [self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab[x] for x in str_tokens]
                    token_to_emb.append({"token_ids":str_tokens_ids, "token":tokens["word"], "index":tokens["index"]})

            
            core_nlp_out_mapping ={}
            input_item_index=0
            n_windows = 3
            for open_nlp_item in token_to_emb:
                found = False
                for open_nlp_item_token_id in open_nlp_item["token_ids"]:
                    upper_limit = input_item_index + n_windows
                    if input_item_index + n_windows > len(input_items):
                        upper_limit = len(input_items)
                    for scan_id in range(input_item_index, upper_limit):
                        if open_nlp_item_token_id == input_items[scan_id]["token_ids"]:
                            if open_nlp_item["index"]-1 not in core_nlp_out_mapping:
                                core_nlp_out_mapping[open_nlp_item["index"]-1] = [scan_id]
                            else:
                                core_nlp_out_mapping[open_nlp_item["index"]-1].append(scan_id)
                            input_item_index=scan_id+1
                            found = True
                            break
                if not found:
                    if input_item_index < len(input_items):
                        core_nlp_out_mapping[open_nlp_item["index"]-1] = [input_item_index]
                    else:
                        core_nlp_out_mapping[open_nlp_item["index"]-1] = [len(input_items)-1]
            if core_nlp_out_mapping[len(core_nlp_out_mapping)-1][-1] != sentence_mask[-1]:
                
                core_nlp_out_mapping_rev = {}
                for key in core_nlp_out_mapping.keys():
                    for value in core_nlp_out_mapping[key]:
                        core_nlp_out_mapping_rev[value] = key
                
                print(core_nlp_out_mapping[len(core_nlp_out_mapping)-1][-1], sentence_mask[-1])
                
                print("------")
                print(sentence)
                print(".")
                print(sentence_mask)
                for index, x in enumerate(sentence): 
                    if index in core_nlp_out_mapping_rev:
                        print(f' {index} {x} - {self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos()[x]} - [{core_nlp_out_mapping_rev[index]}]')
                    else:
                        print(f' {index} {x} - {self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos()[x]} - [NOT EXIST]')

                print("...")
                print(core_nlp_out_mapping)
                
                
                assert("Parsing issue, nodes nbr different")
        return core_nlp_dict , core_nlp_out_mapping

    def get_tag_definition(self, type):
        if type =='POS':
            return self.get_POS_tag_definition()
        elif type == 'Dependency':
            return self.get_dependency_tag_definition()
        elif type =='Constituency':
            return self.get_constituency_tag_definition()
        elif type =='Sequence':
            return self.get_sequence_tag_definition()


    def getData(self, type):
        if type =='POS':
            return self.getPOS()
        elif type == 'Dependency':
            return self.getDependency()
        elif type =='IE':
            return self.getOpenIE()
        elif type =='Constituency':
            return self.getConstituencyData()
        elif type in ('Sequence', "Master", "Amr"):
            return self.getTokenizeSentence()


    def getConstituencyData(self):
        core_nlp_dict,core_nlp_out_mapping = self.sentence_dict,self.sentence_out_mapping
        parsed_results = []
        node_id = 0
        for s_id in range(len(core_nlp_dict["sentences"])):
            tokens = core_nlp_dict["sentences"][s_id]["tokens"]
            token_stack = deque()
            [token_stack.append((index, x['word'])) for index, x in enumerate(tokens)]
            parsed_sentence_data = core_nlp_dict["sentences"][s_id]["parse"]
            for punc in [u"(", u")"]:
                parsed_sentence_data = parsed_sentence_data.replace(punc, " " + punc + " ")
            parse_list = (parsed_sentence_data.strip()).split()

            if parse_list[0] == "(" and parse_list[1] == "ROOT":
                parse_list = parse_list[2:-1]
            for index in range(len(parse_list)):
                if index <= len(parse_list) - 2:
                    if parse_list[index] == "." and parse_list[index + 1] == ".":
                        parse_list[index] = "period"
            stack = deque()
            idx = 0
            nodes = {}
            edges = []
            node_id = len(token_stack)
            while idx < len(parse_list):
                if parse_list[idx] == "(":
                    stack.append(node_id)
                elif parse_list[idx] == ")":
                    parent = stack.pop()
                    if len(stack)>0:
                        edges+=[[stack[-1], parent]]    
                elif idx+1 < len(parse_list):
                    if parse_list[idx+1] not in ["(", ")"]:
                        nodes[node_id] = parse_list[idx]
                        edges+=[[node_id, token_stack.popleft()[0]]]
                        node_id+=1 
                        idx+=1
                    else:
                        nodes[node_id] = parse_list[idx]
                        node_id+=1        
                idx+=1

            parsed_results.append(
                    {"nodes":nodes, "edges":edges, "mapping":core_nlp_out_mapping}
            )
        return parsed_results
    
    
    def getOpenIE(self):

        core_nlp_dict,core_nlp_out_mapping = self.sentence_dict,self.sentence_out_mapping
        parsed_results = []
        node_id = 0
        for s_id in range(len(core_nlp_dict["sentences"])):
            ie_item = []
            for ois in core_nlp_dict["sentences"][s_id]["openie"]:
                sbj = ois["subjectSpan"]
                rel = ois["relationSpan"]
                obj = ois["objectSpan"]
                ie_item.append(
                    {"sub_span": sbj, 
                        "rel_span": rel, 
                        "obj_span": obj})
            parsed_results.append(
                    {"ie_content":ie_item, "mapping":core_nlp_out_mapping}
            )  
        return parsed_results     
    
    
    def getPOS(self):
        core_nlp_dict,core_nlp_out_mapping=self.sentence_dict,self.sentence_out_mapping
        parsed_results = []
        node_id = 0 
        for s_id in range(len(core_nlp_dict["sentences"])):
            parsed_sent = []
            node_item = []
            unique_hash = {}
            node_id = 0

            for tokens in core_nlp_dict["sentences"][s_id]["tokens"]:
                unique_hash[(tokens["index"], tokens["word"])] = node_id
                node = {
                    "token": tokens["word"],
                    "position_id": tokens["index"] - 1,
                    "id": node_id,
                    "sentence_id": s_id,
                    "pos":tokens["pos"]
                }
                node_item.append(node)
                node_id += 1
            parsed_results.append(
                {"node_content": node_item, "node_num": node_id,"mapping":core_nlp_out_mapping}
            )   
        return parsed_results   
    
    def getDependency(self):

        core_nlp_dict,core_nlp_out_mapping = self.sentence_dict,self.sentence_out_mapping
        parsed_results = []
        node_id = 0 
        for s_id in range(len(core_nlp_dict["sentences"])):
            parsed_sent = []
            node_item = []
            unique_hash = {}
            node_id = 0

            for tokens in core_nlp_dict["sentences"][s_id]["tokens"]:
                unique_hash[(tokens["index"], tokens["word"])] = node_id
                node = {
                    "token": tokens["word"],
                    "position_id": tokens["index"] - 1,
                    "id": node_id,
                    "sentence_id": s_id,
                }
                node_item.append(node)
                node_id += 1

            for dep in core_nlp_dict["sentences"][s_id]["basicDependencies"]:

                if dep["governorGloss"] == "ROOT":
                    continue

                if dep["dependentGloss"] == "ROOT":
                    continue

                dep_info = {
                    "edge_type": dep["dep"],
                    "src": unique_hash[(dep["governor"], dep["governorGloss"])],
                    "tgt": unique_hash[(dep["dependent"], dep["dependentGloss"])],
                }
                parsed_sent.append(dep_info)
            parsed_results.append(
                {"graph_content": parsed_sent, "node_content": node_item, "node_num": node_id, "mapping":core_nlp_out_mapping}
            )
        return parsed_results


    def getTokenizeSentence(self, dataset=None,item=None, item_type=None):
        parsed_results = []
        if dataset and item and item_type:
            if dataset and item and item_type:
                self.preprocess_graph(dataset,item, item_type)
                return [self.sentence_dict['sentences'][0]['tokens'][i]["word"] for  i  in range(len(self.sentence_dict['sentences'][0]['tokens']))]
        else:               
            parsed_results.append(
                {"tokens":self.sentence_dict['sentences'][0]['tokens']}
                )
        return parsed_results

    
    
    
    def get_POS_tag_definition(self):
        if self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_embedding_language() == 'en':
            return  [
                ("CC", "Coordinating conjunction"),
                ("CD", "Cardinal number"),
                ("DT", "Determiner"),
                ("EX", "Existential there"),
                ("FW", "Foreign word"),
                ("IN", "Preposition or subordinating conjunction"),
                ("JJ", "Adjective"),
                ("JJR", "Adjective, comparative"),
                ("JJS", "Adjective, superlative"),
                ("LS", "List item marker"),
                ("MD", "Modal"),
                ("NN", "Noun, singular or mass"),
                ("NNS", "Noun, plural"),
                ("NNP", "Proper noun, singular"),
                ("NNPS", "Proper noun, plural"),
                ("PDT", "Predeterminer"),
                ("POS", "Possessive ending"),
                ("PRP", "Personal pronoun"),
                ("PRP$", "Possessive pronoun"),
                ("RB", "Adverb"),
                ("RBR", "Adverb, comparative"),
                ("RBS", "Adverb, superlative"),
                ("RP", "Particle"),
                ("SYM", "Symbol"),
                ("TO", "to"),
                ("UH", "Interjection"),
                ("VB", "Verb, base form"),
                ("VBD", "Verb, past tense"),
                ("VBG", "Verb, gerund or present participle"),
                ("VBN", "Verb, past participle"),
                ("VBP", "Verb, non-3rd person singular present"),
                ("VBZ", "Verb, 3rd person singular present"),
                ("WDT", "Wh-determiner"),
                ("WP", "Wh-pronoun"),
                ("WP$", "Possessive wh-pronoun"),
                ("WRB", "Wh-adverb"), 
                ("." ,  "punctuation"), 
                ("HYPH" ,  "Hyphenation"),
                ("," ,   "punctuation"), 
                (":" ,  "punctuation"),
                ("GW" , "general werb"),
                ("``", "``"),
                ("''", "''"),
                ("AFX", "AFX"),
                ("-LRB-", "-LRB-"),
                ("-RRB-", "-RRB-"),
                ("$", "$"),
                ("ADD", "ADD"),
                ("NFP", "NFP")
                ]
        elif self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_embedding_language() == 'de':
           return [
                ("ADJ", "adjective"),
                ("ADP", "adposition"),
                ("ADV", "adverb"),
                ("AUX", "auxiliary verb"),
                ("CONJ", "coordinating conjunction"),
                ("DET", "determiner"),
                ("INTJ", "interjection"),
                ("NOUN", "noun"),
                ("NUM", "numeral"),
                ("PART", "particle"),
                ("PRON", "pronoun"),
                ("PROPN", "proper noun"),
                ("PUNCT", "punctuation"),
                ("SCONJ", "subordinating conjunction"),
                ("SYM", "symbol"),
                ("VERB", "verb"),
                ("X", "other"),
                ("CCONJ", "CCONJ")
           ]
        else:
            raise Exception('Language not supported')
        
    def get_dependency_tag_definition(self):
        if self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_embedding_language() == 'en':
            return  [
                ("acl","clausal modifier of noun (adjectival clause)"),
                ("advcl","adverbial clause modifier"),
                ("advmod","adverbial modifier"),
                ("amod","adjectival modifier"),
                ("appos","appositional modifier"),
                ("aux","auxiliary"),
                ("auxpass","passive auxiliary"),
                ("case","case marking"),
                ("cc","coordinating conjunction"),
                ("ccomp","clausal complement"),
                ("compound","compound"),
                ("conj","conjunct"),
                ("cop","copula"),
                ("csubj","clausal subject"),
                ("csubjpass","clausal passive subject"),
                ("dep","unspecified dependency"),
                ("det","determiner"),
                ("discourse","discourse element"),
                ("dislocated","dislocated elements"),
                ("dobj","direct object"),
                ("expl","expletive"),
                ("foreign","foreign words"),
                ("goeswith","goes with"),
                ("iobj","indirect object"),
                ("list","list"),
                ("mark","marker"),
                ("mwe","multi-word expression"),
                ("name","name"),
                ("neg","negation modifier"),
                ("nmod","nominal modifier"),
                ("nsubj","nominal subject"),
                ("nsubjpass","passive nominal subject"),
                ("nummod","numeric modifier"),
                ("parataxis","parataxis"),
                ("punct","punctuation"),
                ("remnant","remnant in ellipsis"),
                ("reparandum","overridden disfluency"),
                ("root","root"),
                ("vocative","vocative"),
                ("xcomp","open clausal complement"), 
                ("obl" , "oblique nominal"),
                ("obj" , "object"),
                ("nmod:poss" ,"possessive nominal modifier"),
                ("aux:pass" , "passive auxiliary"),
                ("compound:prt" , "phrasal verb particle"),
                ("acl:relcl" , "relative clause modifier"),
                ("nsubj:pass" , "passive nominal subject"),
                ("obl:tmod" ,  "temporal modifier"),
                ("obl:npmod" ,"noun phrase as adverbial modifier"),
                ("csubj:pass" , "clausal passive subject"),
                ("det:predet", "det:predet"),
                ("cc:preconj", "cc:preconj"),
                ("fixed", "fixed"),
                ("orphan", "orphan") 
            ]
        elif self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_embedding_language() == 'de':
            return  [
                ("acl","clausal modifier of noun (adjectival clause)"),
                ("advcl","adverbial clause modifier"),
                ("advmod","adverbial modifier"),
                ("amod","adjectival modifier"),
                ("appos","appositional modifier"),
                ("aux","auxiliary"),
                ("auxpass","passive auxiliary"),
                ("case","case marking"),
                ("cc","coordinating conjunction"),
                ("ccomp","clausal complement"),
                ("compound","compound"),
                ("conj","conjunct"),
                ("cop","copula"),
                ("csubj","clausal subject"),
                ("csubjpass","clausal passive subject"),
                ("dep","unspecified dependency"),
                ("det","determiner"),
                ("discourse","discourse element"),
                ("dislocated","dislocated elements"),
                ("dobj","direct object"),
                ("expl","expletive"),
                ("foreign","foreign words"),
                ("goeswith","goes with"),
                ("iobj","indirect object"),
                ("list","list"),
                ("mark","marker"),
                ("mwe","multi-word expression"),
                ("name","name"),
                ("neg","negation modifier"),
                ("nmod","nominal modifier"),
                ("nsubj","nominal subject"),
                ("nsubjpass","passive nominal subject"),
                ("nummod","numeric modifier"),
                ("parataxis","parataxis"),
                ("punct","punctuation"),
                ("remnant","remnant in ellipsis"),
                ("reparandum","overridden disfluency"),
                ("root","root"),
                ("vocative","vocative"),
                ("xcomp","open clausal complement"), 
                ("obl" , "oblique nominal"),
                ("obj" , "object"),
                ("nmod:poss" ,"possessive nominal modifier"),
                ("aux:pass" , "passive auxiliary"),
                ("compound:prt" , "phrasal verb particle"),
                ("acl:relcl" , "relative clause modifier"),
                ("nsubj:pass" , "passive nominal subject"),
                ("obl:tmod" ,  "temporal modifier"),
                ("obl:npmod" ,"noun phrase as adverbial modifier"),
                ("csubj:pass" , "clausal passive subject"),
                ("det:predet", "det:predet"),
                ("cc:preconj", "cc:preconj"),
                ("fixed", "fixed"),
                ("orphan", "orphan"), 
                ("det:poss","det:poss"), 
                ("flat", "flat"),
                ("expl:pv", "expl:pv")
            ]    
    
    def get_sequence_tag_definition(self):
        return  [   
                ("seq","Sequence"),
        ]     
        
    def get_constituency_tag_definition(self):
        return  [   
                ("CC","Coordinating conjunction"),
                ("CD","Cardinal number"),
                ("DT","Determiner"),
                ("EX","Existential there"),
                ("FW","Foreign word"),
                ("IN","Preposition or subordinating conjunction"),
                ("JJ","Adjective"),
                ("JJR","Adjective, comparative"),
                ("JJS","Adjective, superlative"),
                ("LS","List item marker"),
                ("MD","Modal"),
                ("NN","Noun, singular or mass"),
                ("NNS","Noun, plural"),
                ("NNP","Proper noun, singular"),
                ("NNPS","Proper noun, plural"),
                ("PDT","Predeterminer"),
                ("POS","Possessive ending"),
                ("PRP","Personal pronoun"),
                ("PRP$","Possessive pronoun"),
                ("RB","	Adverb"),
                ("RBR","Adverb, comparative"),
                ("RBS","Adverb, superlative"),
                ("RP","Particle"),
                ("SYM","Symbol"),
                ("TO","to"),
                ("UH","Interjection"),
                ("VB","Verb, base form"),
                ("VBD","Verb, past tense"),
                ("VBG","Verb, gerund or present participle"),
                ("VBN","Verb, past participle"),
                ("VBP","Verb, non-3rd person singular present"),
                ("VBZ","Verb, 3rd person singular present"),
                ("WDT","Wh-determiner"),
                ("WP","Wh-pronoun"),
                ("WP$","Possessive wh-pronoun"),
                ("WRB","Wh-adverb"),
                ("NP","NP"),
                ("VP","VP"),
                ("PP","PP"),
                ("S","S"),
                ("period","period"),
                ("SBAR","Subordinate Clause"),
                ("WHNP","Wh-noun phrase"),
                ("ADJP","Adjective Phrase"),
                ("ADVP","Adverb Phrase"),
                ("NP-TMP","temporal noun phrase"),
                ("NML","noun-based adjective phrase"),
                (",","punctuation"),
                ("PRT","PRT"),
                ("X","X"),
                ("FRAG","FRAG"),
                ("INTJ","INTJ"),
                ("SQ","SQ"),
                ("WHADVP","WHADVP"),
                ("HYPH","HYPH"),
                ("UPC","UPC"),
                (",","colon"),
                ("WHPP","WHPP"),
                ("PRN","PRN"),
                ("UCP","UCP"),
                ("GW","GW")
            ]


