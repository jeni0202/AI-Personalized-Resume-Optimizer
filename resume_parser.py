import PyPDF2
from docx import Document

def parse_resume(file_path):
    """
    Parse resume from PDF or DOCX file and extract text.
    """
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")
    return text
