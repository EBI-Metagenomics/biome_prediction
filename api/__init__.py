"""A basic API on top of the biome prediction tool"""
import hug

from biome_classifier import load_classifier

biome_classifier = load_classifier.get_model()


@hug.get("/biom-prediction")
def predict(text):
    """Predict the biome based on the provided text
    The output is a list of (biome, score)
    """
    results = biome_classifier.pred_input(text)
    return [{"biome": biome, "score": score} for biome, score in results]
