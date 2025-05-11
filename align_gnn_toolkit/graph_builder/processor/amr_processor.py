from graph_builder.processor.processor_abstract import ProcessorAbstract
import logging
logging.getLogger('penman.layout').setLevel(logging.ERROR)
logging.getLogger('penman.transform').setLevel(logging.ERROR)
import amrlib
from   amrlib.models.parse_xfm.penman_serializer import PenmanSerializer
from amrlib.alignments.rbw_aligner import RBWAligner
from amrlib.graph_processing.annotator import add_lemmas
from penman import surface
from penman.graph import Edge 
from data_set.data_set_processor import DataSetProcessor
import logging

class AMRProcessor(ProcessorAbstract):
    def __init__(self, 
            params=None,
            url=None,
            port=None,
            data_set=None
            ) -> None:
        super(AMRProcessor, self).__init__(params, url, port, data_set)

        self.amr_dict = None
        self.amr_out_mapping = None
        path_to_model = params["amrlib.model_stog"]
        path_to_frame_arm_arg_desc = params["amrlib.propbank-amr-frames"]        
        logging.info(f'Load AMR model [{path_to_model}]')
        self.stog = amrlib.load_stog_model(model_dir=path_to_model)
        logging.info(f'Load propbank amr frames [{path_to_frame_arm_arg_desc}]')
        self.prop_arm_frames =  self.read_frames(path_to_frame_arm_arg_desc)
        logging.getLogger("penmsan.layout").setLevel(logging.WARNING)


    def add_item(self, dic, line):
        line = line.strip()
        #print(line)
        items = line.split("ARG")
        concept = items[0].strip()
        dic[concept] = concept
        dic[concept] = {}
        for item in range(1,len(items)):
            i = items[item].split(":")
            if len(i)>1:
                dic[concept]["ARG"+i[0]] = i[1].strip()


    def read_frames(self, path_to_frames):
        prop_arm_frames = {}
        file1 = open(path_to_frames, 'r', encoding='utf-8')
        prop_list = file1.readlines()
        for line in prop_list:
            self.add_item(prop_arm_frames, line)
        return prop_arm_frames

         
    def getProcessorName(self):
        return "AMR"

    def parseSentence(self, sentence, sentence_mask, sentence_as_string=None):        
        parsed_results = []
        sentence_dict = None
        core_nlp_out_mapping = None
        sentence_as_hash  = None
        if not sentence_as_string:
            sentence_as_string = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_sentence_as_text(sentence)
        if self.isCacheEnabled():
            sentence_as_hash = self.getHashFromSentence(sentence_as_string)
            if self.isInCache(sentence_as_hash):
                parsed_results.append(self.getFromCache(sentence_as_hash))
                return parsed_results
        should_remove_spaces = True # to_do: move to config...
        if should_remove_spaces:
            sentence_as_string = sentence_as_string.strip()
        
        graphs = self.stog.parse_sents([sentence_as_string])
        penman_graph = add_lemmas(graphs[0], snt_key='snt')        
        aligner = RBWAligner.from_penman_w_json(penman_graph)    # use this with an annotated penman graph object
        penaman_graph = PenmanSerializer(aligner.get_graph_string())
        alignments = surface.alignments(penaman_graph.graph)

        output_to_input_mapping = None
        if  sentence_mask and len(aligner.tokens) != sentence_mask[-1]:
            if True:

                str_tokens = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_def_tokenizer(aligner.tokens, is_per_words=True)
                arm_indices = self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).get_tokenizer_indices(aligner.tokens, extract_words=False)
                str_tokens_ids = [self.data_set.data_set_processor.getProcessor(DataSetProcessor.TEXT_PROCESSOR).vocab[x] for x in str_tokens]
                
                output_to_input_mapping = {}
                def get_indices(indices, tokens_ids):
                    output_indices = {}
                    for index, pos in enumerate(indices):
                        if pos not in output_indices:
                            output_indices[pos] = []
                        output_indices[pos].append(tokens_ids[index])
                    return output_indices
                
                output_indices = get_indices(arm_indices, str_tokens_ids)
                input_indices = get_indices(sentence_mask, sentence)
                core_nlp_out_mapping ={}
                last_found_index=0
                n_windows = 3
                for output_key in output_indices.keys():
                    found = False
                    upper_limit = last_found_index + n_windows
                    if last_found_index + n_windows > len(input_indices):
                        upper_limit = len(input_indices)
                    for scan_id in range(last_found_index, upper_limit):
                        if output_indices[output_key] == input_indices[scan_id]:
                            found = True
                            last_found_index = scan_id+1
                            output_to_input_mapping[output_key] = [scan_id]
                            break
                        else:
                            for output_key_item in output_indices[output_key]:
                                if output_key_item in input_indices[scan_id]:
                                    found = True
                                    last_found_index = scan_id+1
                                    output_to_input_mapping[output_key] = [scan_id]
                                    break
                                        
                    if not found:
                        if last_found_index < len(input_indices):
                            output_to_input_mapping[output_key] = [last_found_index-1]
                        else:
                            output_to_input_mapping[output_key] = [len(input_indices)-1]
                
                
                        

        nodes, edges = [], []
        g = penaman_graph.graph
        
        mappings_attributes = {}
        for attrs in penman_graph.attributes():
            if attrs.source not in mappings_attributes:
                mappings_attributes[attrs.source] = []
            mappings_attributes[attrs.source].append((attrs.role,attrs.target))
            
        mapping_org_refiy = {}
        from penman.transform import reify_attributes
        g = reify_attributes(penaman_graph.graph)                                                                    
        index = 0
        instances = g.instances()
        max_index = len(instances)
        while index < max_index:
            if instances[index].source not in mappings_attributes:
                index+=1
            else:
                items = mappings_attributes[instances[index].source]
                for i, item in enumerate(items):
                    mapping_org_refiy[(instances[index].source, item[0])]  = instances[index+1+i].source 
                index+=1+i
        
        def clean_instances(instances):
            out = []
            for instance in instances:
                if instance.role == ':instance' and "-" in instance.target:
                    target = instance.target.replace("-", " ")
                    i= ''.join(ch for ch in target if ch.isalpha() or ch.isspace() )
                    out.append(instance._replace(target=i.strip()))
                else:
                    out.append(instance)
            return out
   
        
        def get_aligned(alignments, mapping_org_refiy):
            aligned = {}
            for key in alignments.keys():
                _key = key[0]
                if (key[0],key[1]) in mapping_org_refiy:
                    _key = mapping_org_refiy[(key[0],key[1])]
                    
                aligned[_key]= alignments[key].indices[0]
            return aligned


        instances = clean_instances(instances)
        aligned = get_aligned(alignments, mapping_org_refiy)

        for instance in instances:
            matched = None
            if instance.source in aligned:
                matched = aligned[instance.source]

            if output_to_input_mapping and matched:
                matched = output_to_input_mapping[matched][0] # pick first from list
            node = {
                "type":instance.role,
                "index": matched,
                "key":instance.source, 
                "name":instance.target
            }
            nodes.append(node) 
            

        _edges = g.edges()
        _instances = g.instances()
        _inst_dic = {}
        for _inst in _instances:
            _inst_dic[_inst.source] = {"role":_inst.role, "target":_inst.target}
            
        for edge in _edges:

            frame = None
            if "ARG" in edge.role: 
                _role = edge.role[1:]
                if "-of" in edge.role: 
                    _frame = _inst_dic[edge.target]["target"]
                    _role =  edge.role.split("-")[0][1:]
                else:
                    _frame = _inst_dic[edge.source]["target"]
                if _frame in self.prop_arm_frames:
                    _frames = self.prop_arm_frames[_frame]
                else:
                    _frames = None
                    logging.warning(f'[{_frame}] not found in propbank-arm')
                if _frames:
                    if _role in _frames:
                        frame = _frames[_role]
                    else:
                        logging.warning(f'[{_role}] not found in frame [{_frame}]')
            
            relation_info = {
                    "edge_type": edge.role,
                    "src": edge.source,
                    "tgt": edge.target,
                    "frame": frame
                }

            edges.append(relation_info)
            

        
        sentence_dict = {"nodes":nodes, "edges":edges, "mapping":core_nlp_out_mapping}
        
        if self.isCacheEnabled():
            self.setToCache(sentence_as_hash, sentence_dict)
        parsed_results.append(sentence_dict)
        return parsed_results
        

    def get_tag_definition(self, type):
        return  [  
                (":ARG","ARG"),
                (":accompanier","accompanier"),
                (":age","age"),
                (":beneficiary","beneficiary"),
                (":cause","cause"),   
                (":concession","concession"),
                (":condition","condition"),
                (":consist-of","consist-of"),
                (":cost","cost"), 
                (":degree","degree"),
                (":destination","destination"),
                (":direction","direction"),
                (":domain","domain"),
                (":duration","duration"),
                (":employed-by","employed-by"),   
                (":example","example"),
                (":extent","extent"), 
                (":frequency","frequency"),
                (":instrument","instrument"),
                (":li","li"),
                (":location","location"),
                (":manner","manner"),
                (":meaning","meaning"),
                (":medium","medium"),
                (":mod","mod"),
                (":mode","mode"),     
                (":name","name"),    
                (":ord","ord"),
                (":part","part"),    
                (":path","path"),   
                (":polarity","polarity"),
                (":polite","polite"),
                (":poss","poss"),   
                (":purpose","purpose"),
                (":role","role"),  
                (":source","source"),
                (":subevent","subevent"),
                (":subset","subset"),
                (":superset","superset"),
                (":time","time"), 
                (":topic","topic"), 
                (":value","value"), 
                (":quant","quant"),  
                (":unit","unit"),
                (":scale","scale"), 
                (":day","day"),
                (":month","month"),  
                (":year","year"),
                (":weekday","weekday"),
                (":timezone","timezone"),
                (":quarter","quarter"),
                (":dayperiod","dayperiod"),
                (":season","season"),
                (":year2","year2"),
                (":decade","decade"),
                (":century","century"),
                (":calendar","calendar"),
                (":era","era"),
                (":op","op[0-9]+"),
                (":snt","snt[0-9]+"),
                (":prep-against","prep-against"), 
                (":prep-along-with","prep-along-with"), 
                (":prep-amid","prep-amid"),
                (":prep-among","prep-among"),
                (":prep-as","prep-as"),
                (":prep-at","prep-at"),
                (":prep-by","prep-by"),
                (":prep-for","prep-for"),
                (":prep-from","prep-from"),
                (":prep-in","prep-in"),
                (":prep-in-addition-to","prep-in-addition-to"),
                (":prep-into","prep-into"),
                (":prep-on","prep-on"),
                (":prep-on-behalf-of","prep-on-behalf-of"),
                (":prep-out-of","prep-out-of"),
                (":prep-to","prep-to"),
                (":prep-toward","prep-toward"),
                (":prep-under","prep-under"),
                (":prep-with","prep-with"),
                (":prep-without","prep-without"),   
                (":conj-as-if","conj-as-if"),
                (":wiki","wiki"),
                (":range","range"),
                (":instance","instance"),
                (":part-of","part-of"), 
                (":ARG-of","ARG-of"),
                (":subevent-of","subevent-of"), 
                (":poss-of","poss-of"), 
                (":instrument-of","instrument-of"), 
                (":location-of","location-of"), 
                (":panda","panda"),
                (":quant-of","quant-of"),
                (":duration-of","duration-of"),
                (":ride-","ride-"),
                (":board-","board-"),
                (",",","),
                (":concession-of","concession-of"),
                (":time-of","time-of"),
                (":extent-of","extent-of"), 
                (":manner-of","manner-of"),
                (":polarity-of","polarity-of"), 
                (":consist","consist"),
                (":topic-of","topic-of"),
                (":",""),
                (":prep-zur", ":prep-zur"), 
                (":example-of", ":example-of"), 
                (":degree-of", ":degree-of"),
                (":beneficiary-of", ":beneficiary-of"), 
                (":path-of", ":path-of"),
                (":type", ":type"),
            ]
    