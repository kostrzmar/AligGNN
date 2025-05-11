from graph_builder.processor.processor_abstract import ProcessorAbstract
from data_set.data_set_processor import DataSetProcessor
import stanza
from collections import deque
from data_set.data_set_abstract import DatasetAbstract


class StanzaProcessor(ProcessorAbstract):
    UNIVERSAL_POS = "UPOS"
    TREE_BANK_POS = "XPOS"   
    
    def __init__(self, 
            params=None,
            url=None,
            port=None,
            data_set=None
            ) -> None:
        super(StanzaProcessor, self).__init__(params, url, port, data_set)

        self.pos_type = StanzaProcessor.UNIVERSAL_POS
        self.stanza = None
        self.stanza_dict = None
        self.stanza_out_mapping = None
        self.stanza_results = None
        self.stanzaAnnotators = ["tokenize","mwt", "lemma"]
        self.annotators = {
            "Constituency":["constituency"], 
            "Dependency":["pos","depparse"],
            "IE":[""], 
            "POS":["pos"]
        }       
           
    def getProcessorName(self):
        return "Stanza"

    def registerAnnotatorsForBuilders(self, builder_names):
        for builder_name in builder_names:
            if builder_name in self.annotators:
                for annotator in self.annotators[builder_name]:
                    if annotator not in self.stanzaAnnotators:
                        self.stanzaAnnotators.append(annotator)
        self.stanza = stanza.Pipeline(lang=self.lang, processors=self.getAnnotators())
        
    def getAnnotators(self):
        return ",".join(self.stanzaAnnotators)

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
            raise Exception(f'IE not supported by [{self.getProcessorName()}]')
            return None
        elif type =='Constituency':
            return self.getConstituencyData()
        elif type in ('Sequence', "Master"):
            return self.getTokenizeSentence()

    def getTokenizeSentence(self, dataset=None,item=None, item_type=None):
        parsed_results = []
        if dataset and item and item_type:
            if dataset and item and item_type:
                self.preprocess_graph(dataset,item, item_type)
                return [self.sentence_dict['sentences'][0]['tokens'][i]["word"] for  i  in range(len(self.sentence_dict['sentences'][0]['tokens']))]
        else:               
            loc_tokens = self.sentence_dict[0]
            for token in loc_tokens:
                token["word"] = token["text"]        
            parsed_results.append(
                {"tokens":loc_tokens}
                )
        return parsed_results

    def parse_stanza(self, sentence, concatenate_results=True):
        stanza_results = self.stanza(sentence)
        stanza_results = stanza_results.to_dict()
        if concatenate_results and len(stanza_results)>1:
            stanza_results_concat = []
            stanza_results_concat.append(stanza_results[0])
            last_id = stanza_results_concat[0][-1]['id']
            last_char = stanza_results_concat[0][-1]['end_char']
            for sent_id in range(1, len(stanza_results)):
                items = stanza_results[sent_id]  
                for index, item in enumerate(items):
                    if type(item["id"]) is tuple:
                        item['id'] = (item['id'][0]+last_id,item['id'][1]+last_id)
                    else:    
                        item['id'] = item['id']+last_id
                    if 'head' in item:
                        item['head'] = item['head']+last_id
                    if 'end_char' in item:
                        item['end_char'] = item['end_char']+last_char
                stanza_results_concat[0].extend(items)
                last_id = item['id']
                last_char = item['end_char'] 
            return stanza_results_concat
        return stanza_results
            
        
        
    def preprocess_graph(self, dataset,item, item_type): 
        stanza_results = self.parse_stanza(item[dataset.get_token_name(item_type,DatasetAbstract.SENTENCE_TOKEN_NAME)])
        self.sentence_dict = stanza_results

    def parseSentence(self, sentence, sentence_mask):        
        stanza_out_mapping = None
        stanza_dict = None
        stanza_results = None
        sentence_as_hash = None
        sentence_as_string = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_sentence_as_text(sentence)
        if self.isCacheEnabled():
            sentence_as_hash = self.getHashFromSentence(sentence_as_string, "_".join(str(i) for i in self.stanzaAnnotators))
            if self.isInCache(sentence_as_hash):
                stanza_results = self.getFromCache(sentence_as_hash)
                stanza_dict = stanza_results.to_dict()
                #clean up tuple
                for s_id in range(len(stanza_dict)):
                    for index, tokens in enumerate(stanza_dict[s_id]):
                        if type(tokens["id"]) is tuple:  
                            del stanza_dict[s_id][index]
        
        if not stanza_dict:
            stanza_results = self.parse_stanza(sentence_as_string.strip())
            stanza_dict = stanza_results
            for s_id in range(len(stanza_dict)):
                for index, tokens in enumerate(stanza_dict[s_id]):
                    if type(tokens["id"]) is tuple:  
                        del stanza_dict[s_id][index]
           
            
            if self.isCacheEnabled():
                self.setToCache(sentence_as_hash, stanza_results)  
            

        if  len(stanza_dict[0]) != len(sentence_mask):
            stanza_out_mapping = {}
            token_to_emb=[]
            input_items = []
            for item in sentence:
                input_items.append({"token_ids":item, "token":self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab.get_itos()[item]})
                
            for s_id in range(len(stanza_dict)):
                for tokens in stanza_dict[s_id]:
                    str_tokens = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_tokenizer(tokens["text"])
                    str_tokens_ids = [self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab[x] for x in str_tokens]
                    token_to_emb.append({"token_ids":str_tokens_ids, "token":tokens["text"], "index":tokens["id"]})

            def get_alphanumeric(string):
                only_alphanumeric = ''.join(char for char in string if char.isalnum())
                if len(only_alphanumeric) >0:
                    return only_alphanumeric
                return string
            
            def scan_for_match(input_items, token_to_emb, input_token_index, output_token_index):
                n_windows = 4
                upper_input_limit = [input_token_index + n_windows,  len(input_items)][input_token_index + n_windows > len(input_items)] 
                upper_output_limit = [output_token_index + n_windows,  len(token_to_emb)][output_token_index + n_windows > len(token_to_emb)] 
                for input_scan_id in range(input_token_index, upper_input_limit):
                    for output_scan_id in range(output_token_index, upper_output_limit):
                        for token_scan_emb_id in token_to_emb[output_scan_id]["token_ids"]:
                            if token_scan_emb_id == input_items[input_scan_id]["token_ids"]:
                                return input_scan_id, output_scan_id
                            else:
                                input_token = get_alphanumeric(input_items[input_scan_id]["token"])
                                output_token = get_alphanumeric(token_to_emb[output_scan_id]["token"])
                                if input_token!='' and output_token!='' and input_token in output_token or output_token in input_token:
                                    return input_scan_id, output_scan_id
                                
                return -1, -1
            
            output_to_input_mapping = None
            
            if len(token_to_emb) > len(input_items):
                output_to_input_mapping = {}
                input_token_index = 0
                output_token_index = 0
                while output_token_index < len(token_to_emb):
                    input_token_index = [input_token_index,  len(input_items)-1][input_token_index >= len(input_items)] 
                    token = token_to_emb[output_token_index]
                    if output_token_index not in output_to_input_mapping:
                        output_to_input_mapping[output_token_index] = []  
                    for token_emb_id in token["token_ids"]:
                        if token_emb_id == input_items[input_token_index]["token_ids"]:
                            output_to_input_mapping[output_token_index].append(input_token_index)
                            input_token_index+=1
                        else:
                            
                            input_token = get_alphanumeric(input_items[input_token_index]["token"])
                            output_token = get_alphanumeric(token["token"])
                            if input_token in output_token or output_token in input_token:
                                output_to_input_mapping[output_token_index].append(input_token_index)
                            else:
                                input_scan_id, output_scan_id  = scan_for_match(input_items, token_to_emb, input_token_index, output_token_index)
                                if input_scan_id >= 0 and output_scan_id >=0:
                                    for index in range(output_token_index, output_scan_id):
                                        output_to_input_mapping[index] = [input_scan_id-1]
                                    output_to_input_mapping[output_scan_id] = [input_scan_id]
                                    output_token_index=output_scan_id
                                    input_token_index=input_scan_id+1
                                else:
                                    print("NOT FOUND !!!!")

                    output_token_index+=1
            else:
                print("!!!output hight then input!!!!")

            stanza_out_mapping ={}
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
                            if open_nlp_item["index"]-1 not in stanza_out_mapping:
                                stanza_out_mapping[open_nlp_item["index"]-1] = [scan_id]
                            else:
                                stanza_out_mapping[open_nlp_item["index"]-1].append(scan_id)
                            input_item_index=scan_id+1
                            found = True
                            break
                if not found:
                    if input_item_index < len(input_items):
                        stanza_out_mapping[open_nlp_item["index"]-1] = [input_item_index]
                    else:
                        stanza_out_mapping[open_nlp_item["index"]-1] = [len(input_items)-1]
                        
            stanza_out_mapping = output_to_input_mapping
        return stanza_dict , stanza_out_mapping, stanza_results


    def getPOS(self):
        stanza_dict,stanza_mapping=self.sentence_dict,self.sentence_out_mapping
        parsed_results = []
        node_id = 0 
        for s_id, sentence in enumerate(stanza_dict):
            node_item = []
            node_id = 0

            for token in sentence:
                if type(token["id"]) is tuple:
                    continue
                pos = ""
                if self.pos_type == StanzaProcessor.UNIVERSAL_POS:
                    pos = token["upos"]
                else:
                    pos = token["xpos"]
                node = {
                            "token": token["text"],
                            "position_id": token["id"] - 1,
                            "id": node_id,
                            "sentence_id": s_id,
                            "pos":pos
                        }
                node_item.append(node)
                node_id += 1
            parsed_results.append(
                {"node_content": node_item, "node_num": node_id,"mapping":stanza_mapping}
            )   
        return parsed_results
     
     
    def getDependency(self):
        stanza_dict,stanza_mapping = self.sentence_dict,self.sentence_out_mapping
        parsed_results = []
        node_id = 0 
        for s_id, sentence in enumerate(stanza_dict):
            parsed_sent = []
            node_item = []
            unique_hash = {}
            node_id = 0
            dep_id_2_word ={}
            for node_id, token in enumerate(sentence):
                if type(token["id"]) is tuple:
                    continue
                unique_hash[(token["id"]-1, token["text"])] = node_id
                dep_id_2_word[token["id"]-1] = token["text"]
            
            for token in sentence:
                if type(token["id"]) is tuple:
                    continue
                node = {
                    "token": token["text"],
                    "position_id": token["id"] - 1,
                    "id": node_id,
                    "sentence_id": s_id,
                }
                node_item.append(node)
                node_id += 1
                if token["head"] != 0:
                    dep_info = {
                        "edge_type": token["deprel"],
                        "src": unique_hash[(token["id"]-1, token["text"])],
                        "tgt": unique_hash[(token["head"]-1, dep_id_2_word[token["head"]-1])],
                    }
                    parsed_sent.append(dep_info)
            parsed_results.append(
                {"graph_content": parsed_sent, "node_content": node_item, "node_num": node_id, "mapping":stanza_mapping}
            )
        return parsed_results       
    
    def getConstituencyData(self):
        stanza_dict,stanza_mapping, stanza_results = self.sentence_dict,self.sentence_out_mapping, self.stanza_results
        parsed_results = []
        node_id = 0
        for s_id, sentence in enumerate(stanza_dict):
            tokens = stanza_results.sentences[s_id].words
            token_stack = deque()
            [token_stack.append((index, x.text)) for index, x in enumerate(tokens)]
            parsed_sentence_data = stanza_results.sentences[s_id].constituency.pretty_print()
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
                    {"nodes":nodes, "edges":edges, "mapping":stanza_mapping}
            )
        return parsed_results
    
    
    
    def get_POS_tag_definition(self):
        if self.pos_type == StanzaProcessor.UNIVERSAL_POS:
            return self.get_Universal_POS_tag_definition()
        elif self.pos_type == StanzaProcessor.TREE_BANK_POS:
            language = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_embedding_language()
            if language  == 'en':
                return self.get_en_POS_tag_definition()
            else:
                return self.get_Universal_POS_tag_definition()


   
    def get_Universal_POS_tag_definition(self):
        return [
                ("ADJ", "adjective"),
                ("ADP", "adposition"),
                ("ADV", "adverb"),
                ("AUX", "auxiliary verb"),
                ("CCONJ", "coordinating conjunction"),
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
                ("X", "other",)
           ]
        
    def get_en_POS_tag_definition(self):   
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
        
    def get_dependency_tag_definition(self):
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
                ("obl:agent", "obl:agent"),
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
                ("flat", "flat"),
                ("advcl:relcl","advcl:relcl"),
                ("nsubj:outer","nsubj:outer"),
                ("det:poss","det:poss"),
                ("expl:pv", "expl:pv")
                
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
                ("." ,  "punctuation"), 
                (":" ,  "punctuation"),
                ("``", "d-quote"),
                ("''", "d-quote"),
                ("WHPP", "WHPP")
            ]
        
    def get_sequence_tag_definition(self):
        return  [   
                ("seq","Sequence"),
        ]     