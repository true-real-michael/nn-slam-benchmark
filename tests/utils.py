import aero_vloc as avl

from pathlib import Path

salad = avl.SALAD()
light_glue = avl.LightGlue()
database = avl.Data(Path("tests"), "test_data")
queries = avl.Queries(Path("tests"), "test_data", knn=database.knn)


def create_localization_pipeline():
    """
    Creates localization pipeline based on SALAD place recognition system,
    LightGlue keypoint matcher and test satellite map
    """
    faiss_searcher = avl.FaissSearcher()
    retrieval_system = avl.RetrievalSystem(salad, database, light_glue, faiss_searcher)
    localization_pipeline = avl.LocalizationPipeline(
        retrieval_system
    )
    return localization_pipeline
