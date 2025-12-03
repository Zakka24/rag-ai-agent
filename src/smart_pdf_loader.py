# src/smart_pdf_loader.py

from typing import List, Optional
from pathlib import Path

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

from langchain_core.documents import Document  # o: from langchain.schema import Document


class SmartPDFLoader:
    """
    Loader che:
    - usa PyMuPDF per leggere il testo digitale dalle pagine
    - se una pagina non ha testo (tipico scan), fa OCR full-page con Tesseract
    In questo modo:
      - eviti completely UnstructuredPDFLoader (e il suo bug)
      - gestisci anche le scansioni e le scritte a mano (nei limiti dell'OCR).
    """

    def __init__(self, file_path: str, lang: str = "ita"):
        self.file_path = str(file_path)
        self.lang = lang

    def load(self) -> List[Document]:
        pdf_path = Path(self.file_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        pdf = fitz.open(self.file_path)
        num_pages = len(pdf)

        result_docs: List[Document] = []

        for page_index in range(num_pages):
            page_num = page_index + 1
            page = pdf[page_index]

            # 1) Prova a prendere il testo "normale" (digitale) dalla pagina
            text = page.get_text().strip()

            if text:
                # Pagina con testo digitale: creiamo un Document
                result_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "ocr": False,
                        },
                    )
                )
            else:
                # Pagina "muta" -> probabilmente è una scansione: usa OCR full-page
                ocr_doc = self._ocr_page(pdf_path, page_num)
                if ocr_doc:
                    result_docs.append(ocr_doc)

        pdf.close()
        return result_docs

    def _ocr_page(self, pdf_path: Path, page_num: int) -> Optional[Document]:
        """
        OCR full-page:
        - converte SOLO la pagina richiesta in immagine
        - usa pytesseract per leggere tutto (testo stampato + scritto a mano se riconoscibile)
        """
        images = convert_from_path(
            str(pdf_path),
            dpi=300,
            first_page=page_num,
            last_page=page_num,
        )
        if not images:
            return None

        image = images[0]

        # OCR generale (senza whitelist aggressiva) così prende qualsiasi campo
        text = pytesseract.image_to_string(image, lang=self.lang)
        text = text.strip()
        if not text:
            return None

        return Document(
            page_content=text,
            metadata={
                "source": pdf_path.name,
                "page": page_num,
                "ocr": True,
            },
        )
