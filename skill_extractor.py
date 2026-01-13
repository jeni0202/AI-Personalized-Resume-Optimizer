import spacy
from typing import List, Set
import re

class SkillExtractor:
    """
    A class to extract skills from text using NLP techniques with spaCy.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        """
        Initialize the SkillExtractor with a spaCy model.

        Args:
            model (str): The spaCy model to use. Defaults to 'en_core_web_sm'.
        """
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Model '{model}' not found. Please install it using: python -m spacy download {model}")
            raise

        # Common skill keywords (can be expanded)
        self.skill_keywords = {
            'programming languages': {'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust', 'swift', 'kotlin'},
            'frameworks': {'django', 'flask', 'react', 'angular', 'vue', 'spring', 'hibernate', 'tensorflow', 'pytorch', 'scikit-learn'},
            'databases': {'mysql', 'postgresql', 'mongodb', 'redis', 'sqlite', 'oracle', 'sql server'},
            'tools': {'git', 'docker', 'kubernetes', 'jenkins', 'aws', 'azure', 'gcp', 'linux', 'windows'},
            'soft skills': {'communication', 'leadership', 'teamwork', 'problem solving', 'analytical thinking'},
            'methodologies': {'agile', 'scrum', 'kanban', 'waterfall', 'devops', 'ci/cd'}
        }

        # Flatten all skills into a single set for quick lookup
        self.all_skills = set()
        for category in self.skill_keywords.values():
            self.all_skills.update(category)

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from the given text.

        Args:
            text (str): The text to extract skills from.

        Returns:
            List[str]: A list of extracted skills.
        """
        if not text:
            return []

        # Preprocess text: lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())

        doc = self.nlp(text)
        extracted_skills = set()

        # Extract skills based on keyword matching
        words = set(text.split())
        for skill in self.all_skills:
            if skill in words or skill.replace(' ', '') in text:
                extracted_skills.add(skill)

        # Extract potential skills using POS tagging (nouns and proper nouns)
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN'] and len(token.text) > 2:
                # Check if it's a potential skill (not common stop words)
                if not token.is_stop and token.text not in ['experience', 'skills', 'knowledge', 'ability', 'years']:
                    # Look for compound skills (e.g., "machine learning")
                    if token.dep_ == 'compound' and token.head.pos_ in ['NOUN', 'PROPN']:
                        skill = f"{token.text} {token.head.text}"
                        if skill in self.all_skills or len(skill.split()) > 1:
                            extracted_skills.add(skill.lower())
                    else:
                        extracted_skills.add(token.text.lower())

        # Filter out very common words that are not skills
        filtered_skills = []
        for skill in extracted_skills:
            if len(skill) > 2 and skill not in ['work', 'team', 'project', 'company', 'client', 'user', 'system', 'data', 'time', 'process']:
                filtered_skills.append(skill)

        return sorted(list(set(filtered_skills)))

    def categorize_skills(self, skills: List[str]) -> dict:
        """
        Categorize extracted skills into predefined categories.

        Args:
            skills (List[str]): List of skills to categorize.

        Returns:
            dict: Dictionary with categories as keys and lists of skills as values.
        """
        categorized = {category: [] for category in self.skill_keywords.keys()}

        for skill in skills:
            for category, skill_set in self.skill_keywords.items():
                if skill in skill_set:
                    categorized[category].append(skill)
                    break

        # Remove empty categories
        categorized = {k: v for k, v in categorized.items() if v}

        return categorized

# Example usage
if __name__ == "__main__":
    extractor = SkillExtractor()

    sample_text = """
    I am a software engineer with experience in Python, Java, and JavaScript.
    I have worked with Django, React, and TensorFlow. I am proficient in MySQL and MongoDB.
    I have strong communication skills and experience with Agile methodologies.
    """

    skills = extractor.extract_skills(sample_text)
    print("Extracted Skills:", skills)

    categorized = extractor.categorize_skills(skills)
    print("Categorized Skills:", categorized)
