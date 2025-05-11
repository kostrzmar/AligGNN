from engine import AbstractEngineFactory
from engine import AbstractEngine
from engine.impl import ExperimentEngine

class ExperimentEngineFactory(AbstractEngineFactory):

    def getEngine(self) -> AbstractEngine:
        return ExperimentEngine()