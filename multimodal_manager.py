import sys

# STEP 1: CREATE MOCK OBJECTS FOR THE MISSING LANGCHAIN ATTRIBUTES
class MockAnnotation:
    pass

# STEP 2: PRECISION INJECTION INTO TARGET NAMESPACES
target_modules = ['typing', 'langchain_core.load.serializable', 'pydantic.main']

for mod_name in target_modules:
    if mod_name in sys.modules:
        mod = sys.modules[mod_name]
        setattr(mod, 'BaseCache', MockAnnotation)
        setattr(mod, 'Callbacks', MockAnnotation)
    else:
        import types
        mod = types.ModuleType(mod_name)
        setattr(mod, 'BaseCache', MockAnnotation)
        setattr(mod, 'Callbacks', MockAnnotation)

globals()['BaseCache'] = MockAnnotation
globals()['Callbacks'] = MockAnnotation

# STEP 3: SAFELY IMPORT LIBRARIES
import numpy as np
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from PIL import Image
import config

from langchain_openai import ChatOpenAI

try:
    ChatOpenAI.model_rebuild()
except Exception:
    pass

# Define structured schema output for final retrieval verification
class RerankedResult(BaseModel):
    restaurant_name: str
    combined_score: float
    match_type: str
    cuisine: str

class MultimodalVectorEngine:
    def __init__(self):
        self.text_vector_database: List[Dict[str, Any]] = []
        self.image_vector_database: List[Dict[str, Any]] = []
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=config.OPENAI_API_KEY)

    def generate_text_embedding(self, text: str) -> np.ndarray:
        seed = sum(ord(char) for char in text) % 1000
        np.random.seed(seed)
        vec = np.random.randn(512)
        return vec / np.linalg.norm(vec)

    def generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        width, height = image.size
        seed = (width * height) % 1000
        np.random.seed(seed)
        vec = np.random.randn(512)
        return vec / np.linalg.norm(vec)

    def calculate_cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def index_restaurant_profile(self, name: str, cuisine: str, menu_text: str, menu_image: Optional[Image.Image] = None):
        text_vector = self.generate_text_embedding(menu_text)
        metadata = {"restaurant_name": name, "cuisine_type": cuisine.lower()}
        
        self.text_vector_database.append({
            "vector": text_vector,
            "metadata": metadata,
            "raw_content": menu_text
        })
        
        if menu_image:
            image_vector = self.generate_image_embedding(menu_image)
            self.image_vector_database.append({
                "vector": image_vector,
                "metadata": metadata
            })

    def retrieve_and_fuse(self, 
                          query_text: str, 
                          query_image: Optional[Image.Image] = None, 
                          text_weight: float = 0.5, 
                          metadata_filter: Optional[Dict[str, Any]] = None) -> List[RerankedResult]:
        query_text_vector = self.generate_text_embedding(query_text)
        query_image_vector = self.generate_image_embedding(query_image) if query_image else None
        
        scores_by_restaurant: Dict[str, Dict[str, Any]] = {}

        for item in self.text_vector_database:
            meta = item["metadata"]
            if metadata_filter:
                match = all(meta.get(k) == str(v).lower() for k, v in metadata_filter.items())
                if not match:
                    continue
            
            r_name = meta["restaurant_name"]
            sim = self.calculate_cosine_similarity(query_text_vector, item["vector"])
            
            scores_by_restaurant[r_name] = {
                "text_score": sim,
                "image_score": 0.0,
                "cuisine": meta["cuisine_type"]
            }

        # FIXED BY EXPLICIT NOT-NONE CHECKING BELOW
        if query_image_vector is not None:
            for item in self.image_vector_database:
                meta = item["metadata"]
                if metadata_filter:
                    match = all(meta.get(k) == str(v).lower() for k, v in metadata_filter.items())
                    if not match:
                        continue
                        
                r_name = meta["restaurant_name"]
                sim = self.calculate_cosine_similarity(query_image_vector, item["vector"])
                if r_name in scores_by_restaurant:
                    scores_by_restaurant[r_name]["image_score"] = sim

        image_weight = 1.0 - text_weight
        reranked_list = []
        
        for r_name, scores in scores_by_restaurant.items():
            t_score = scores["text_score"]
            i_score = scores["image_score"]
            
            # FIXED BY EXPLICIT NOT-NONE CHECKING BELOW
            if query_image_vector is not None and i_score > 0.0:
                combined_score = (t_score * text_weight) + (i_score * image_weight)
                match_type = "Cross-Modal (Text + Image Fusion)"
            else:
                combined_score = t_score
                match_type = "Text Semantic Match"

            reranked_list.append(RerankedResult(
                restaurant_name=r_name,
                combined_score=round(combined_score, 4),
                match_type=match_type,
                cuisine=scores["cuisine"].capitalize()
            ))

        return sorted(reranked_list, key=lambda x: x.combined_score, reverse=True)
