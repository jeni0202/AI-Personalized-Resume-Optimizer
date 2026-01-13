from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple

class SimilarityComparator:
    """
    A class to compare resumes and job descriptions using Sentence-BERT embeddings and cosine similarity.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the SimilarityComparator with a Sentence-BERT model.

        Args:
            model_name (str): The name of the Sentence-BERT model to use.
        """
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Error loading model '{model_name}': {e}")
            print("Please ensure sentence-transformers is installed: pip install sentence-transformers")
            raise

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Args:
            texts (List[str]): List of texts to encode.

        Returns:
            np.ndarray: Array of embeddings.
        """
        return self.model.encode(texts)

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1 (str): First text.
            text2 (str): Second text.

        Returns:
            float: Cosine similarity score between 0 and 1.
        """
        embeddings = self.encode_texts([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)

    def compare_resume_to_jd(self, resume_text: str, jd_text: str) -> Dict[str, float]:
        """
        Compare a resume to a job description and return detailed similarity scores.

        Args:
            resume_text (str): The resume text.
            jd_text (str): The job description text.

        Returns:
            Dict[str, float]: Dictionary containing overall similarity and component similarities.
        """
        # Split texts into sections (simple approach: split by double newlines)
        resume_sections = [s.strip() for s in resume_text.split('\n\n') if s.strip()]
        jd_sections = [s.strip() for s in jd_text.split('\n\n') if s.strip()]

        # Compute overall similarity
        overall_similarity = self.compute_similarity(resume_text, jd_text)

        # Compute section-wise similarities
        section_similarities = []
        for resume_section in resume_sections:
            for jd_section in jd_sections:
                sim = self.compute_similarity(resume_section, jd_section)
                section_similarities.append(sim)

        avg_section_similarity = np.mean(section_similarities) if section_similarities else 0.0

        return {
            'overall_similarity': overall_similarity,
            'avg_section_similarity': avg_section_similarity,
            'max_section_similarity': max(section_similarities) if section_similarities else 0.0,
            'min_section_similarity': min(section_similarities) if section_similarities else 0.0
        }

    def find_best_matches(self, resume_text: str, jd_list: List[str], top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the best matching job descriptions for a given resume.

        Args:
            resume_text (str): The resume text.
            jd_list (List[str]): List of job description texts.
            top_k (int): Number of top matches to return.

        Returns:
            List[Tuple[int, float]]: List of tuples (index, similarity_score) for top matches.
        """
        similarities = []
        for i, jd_text in enumerate(jd_list):
            sim = self.compute_similarity(resume_text, jd_text)
            similarities.append((i, sim))

        # Sort by similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_similarity_score(self, similarity: float) -> str:
        """
        Convert similarity score to a qualitative assessment.

        Args:
            similarity (float): Similarity score between 0 and 1.

        Returns:
            str: Qualitative assessment.
        """
        if similarity >= 0.8:
            return "Excellent match"
        elif similarity >= 0.6:
            return "Good match"
        elif similarity >= 0.4:
            return "Fair match"
        elif similarity >= 0.2:
            return "Poor match"
        else:
            return "Very poor match"

# Example usage
if __name__ == "__main__":
    comparator = SimilarityComparator()

    sample_resume = """
    Software Engineer with 5 years of experience in Python, Java, and web development.
    Proficient in Django, React, and AWS. Strong background in data structures and algorithms.
    """

    sample_jd = """
    We are looking for a Software Engineer with experience in Python, JavaScript, and cloud technologies.
    Knowledge of React, Django, and AWS is a plus. Familiarity with agile methodologies required.
    """

    similarity_score = comparator.compute_similarity(sample_resume, sample_jd)
    print(f"Similarity Score: {similarity_score:.4f}")
    print(f"Assessment: {comparator.get_similarity_score(similarity_score)}")

    detailed_comparison = comparator.compare_resume_to_jd(sample_resume, sample_jd)
    print("Detailed Comparison:")
    for key, value in detailed_comparison.items():
        print(f"  {key}: {value:.4f}")
