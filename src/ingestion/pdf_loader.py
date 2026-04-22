from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)

    return texts

if __name__ == "__main__":
    file_path = r"./././data/raw/fastapi-contrib-readthedocs-io-en-latest.pdf"
    texts = load_pdf(file_path)

    print(texts)
    