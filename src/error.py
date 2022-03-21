from src.dataset.loader import Loader
from src.util.detector import Detector
from libs.datil.flag import Flag
from src.util import config
class Error:
    @staticmethod
    def run():
        loader = Loader()
        actual_sentences = loader.get_actual_sentences()
        predicted_sentences = loader.get_w2v_predictions()
        detector = Detector()
        detector.train(actual_sentences=actual_sentences, predicted_sentences=predicted_sentences, order=3, model_path=config.MAPPING_MODEL_PATH)
        
