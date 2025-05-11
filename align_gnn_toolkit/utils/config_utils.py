import os
import logging
from configparser import ConfigParser, ExtendedInterpolation
import yaml
import pathlib
from ray import tune


class ConfigUtils():

    def __init__(self, pathToConfiguration, config_as_dict=None, path_to_template_file=None):        
        if not config_as_dict:
            assert pathToConfiguration, "Configuration can't be found"
            suffix = pathlib.Path(pathToConfiguration).suffix
            self.config_as_dict = None
            self.template_as_dict = None
            self.multiple_config = None
            if suffix == '.init':
                self.config_as_dict = self.loadConfigParser(pathToConfiguration)    
            elif suffix =='.yaml':
                self.config_as_dict = self.loadYaml(pathToConfiguration)   
                if path_to_template_file:
                    self.template_as_dict =  self.loadYaml(path_to_template_file)  
                    assert self.template_as_dict, "Template initialization failed" 
            assert self.config_as_dict, "Configuration initialization failed"
        else:
            self.config_as_dict = config_as_dict
        super().__init__() 

    def extractDataFromConfig(self, configParser: ConfigParser, sectionName):
        section_as_dict = {}
        options = configParser.options(sectionName)
        for option in options:
            section_as_dict[option]  = configParser.get(sectionName,option)
        return section_as_dict


    def loadConfigParser(self, pathToConfiguration):
        config_as_dict = {}
        config_parser =  ConfigParser(interpolation=ExtendedInterpolation())
        if not config_parser.read(pathToConfiguration):
                logging.error("Config file [{}] not found".format(pathToConfiguration))
                raise FileNotFoundError(pathToConfiguration) 
        sections = config_parser.sections()
        for section in sections:
            config_as_dict[section]=self.extractDataFromConfig(config_parser, section)

        defaults = config_parser.defaults()
        default_dict = {}
        for key in defaults:
            default_dict[key] = defaults[key]
        config_as_dict["default_section"] = default_dict
        return config_as_dict
    
    def loadYaml(self, pathToConfiguration):
        multiple_config = []
        try:
            with open(pathToConfiguration, "r") as fh:
                multiple_config = list(yaml.load_all(fh, Loader=yaml.SafeLoader))
        except yaml.YAMLError as exc:
            logging.error("Error during parsing yaml file")
            return None
        if self.multiple_config:
            self.multiple_config.extend(multiple_config)
        else:
            self.multiple_config = multiple_config
        return self.multiple_config[-1]

    def setValue(self, sectionName, propertyName, value):
        for config in self.multiple_config:                
            config[sectionName][propertyName] = value

    def _get_value(self, sectionName, propertyName, defaultValue=None):
        
        template_value=None
        if self.template_as_dict:
            if sectionName in self.template_as_dict:
                if propertyName in self.template_as_dict[sectionName]:
                    template_value = self.template_as_dict[sectionName][propertyName]
            else:
                if propertyName in self.template_as_dict:
                    template_value =  self.template_as_dict[propertyName]
            
        if sectionName in self.config_as_dict:
            if propertyName in self.config_as_dict[sectionName]:
                return self.config_as_dict[sectionName][propertyName]
            else:
                if template_value or isinstance(template_value, bool):
                    return template_value
                return defaultValue  
        else:
            if propertyName in self.config_as_dict:
                return self.config_as_dict[propertyName]
            else:
                if template_value or isinstance(template_value, bool):
                    return template_value
                return defaultValue

    def getValue(self, sectionName, propertyName, defaultValue=None, ray_fine_tune=False):
        if ray_fine_tune:
            ray_propertyName = propertyName+"_ray"
            if ray_propertyName in self.config_as_dict[sectionName]:
                ray_property_value =  self.config_as_dict[sectionName][ray_propertyName]
                if ray_property_value and ray_property_value[0] == 'tune.choice':
                    return tune.choice(ray_property_value[1])
                elif ray_property_value and ray_property_value[0] == "tune.loguniform":
                    return tune.loguniform(ray_property_value[1], ray_property_value[2])  
                    
        return self._get_value(sectionName, propertyName, defaultValue)         
    
    def hasMultipleConfiguration(self):
        if self.template_as_dict:
            return self.multiple_config and len(self.multiple_config)>2
        return self.multiple_config and len(self.multiple_config)>1

    def getNumberOfConfiguration(self):
        return len(self.multiple_config)

    def setActiveConfiguration(self, index):
        assert index < self.getNumberOfConfiguration(), "No configuration for the index->"+str(index)
        self.config_as_dict = self.multiple_config[index]
        
    def convertDictToYaml(self, dict):
        return yaml.dump(dict)