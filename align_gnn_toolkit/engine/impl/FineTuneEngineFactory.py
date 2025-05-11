from engine import AbstractEngineFactory
from engine import AbstractEngine
from engine.impl import FineTuneEngine

class FineTuneEngineFactory(AbstractEngineFactory):

    def getEngine(self) -> AbstractEngine:
        return FineTuneEngine()