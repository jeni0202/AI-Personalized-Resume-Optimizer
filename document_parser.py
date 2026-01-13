import os
from typing import Optional
import PyPDF2
from docx import Document

class DocumentParser:
    """
    A class to parse PDF and DOCX documents into plain text.
    """

    @staticmethod
    def parse_pdf(file_path: str) -> Optional[str]:
        """
        Parse a PDF file and extract text.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            Optional[str]: Extracted text or None if parsing fails.
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error parsing PDF {file_path}: {e}")
            return None

    @staticmethod
    def parse_docx(file_path: str) -> Optional[str]:
        """
        Parse a DOCX file and extract text.

        Args:
            file_path (str): Path to the DOCX file.

        Returns:
            Optional[str]: Extracted text or None if parsing fails.
        """
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Error parsing DOCX {file_path}: {e}")
            return None

    @staticmethod
    def parse_document(file_path: str) -> Optional[str]:
        """
        Parse a document (PDF or DOCX) and extract text.

        Args:
            file_path (str): Path to the document file.

        Returns:
            Optional[str]: Extracted text or None if parsing fails.
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            return None

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            return DocumentParser.parse_pdf(file_path)
        elif file_extension == '.docx':
            return DocumentParser.parse_docx(file_path)
        else:
            print(f"Unsupported file type: {file_extension}")
            return None

# Example usage
if __name__ == "__main__":
    # Test with sample files (assuming they exist)
    pdf_text = DocumentParser.parse_document("sample_resume.pdf")
    docx_text = DocumentParser.parse_document("sample_jd.docx")

    if pdf_text:
        print("PDF Text Extracted:")
        print(pdf_text[:500] + "..." if len(pdf_text) > 500 else pdf_text)

    if docx_text:
        print("\nDOCX Text Extracted:")
        print(docx_text[:500] + "..." if len(docx_text) > 500 else docx_text)
