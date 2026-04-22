def chunk_text(texts, chunk_size=300, overlap=50):
    chunks = []

    for text in texts:
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)

    return chunks

if __name__ == "__main__":
    from pdf_loader import load_pdf

    file_path = r"./././data/raw/fastapi-contrib-readthedocs-io-en-latest.pdf"
    texts = load_pdf(file_path)

    chunks = chunk_text(texts)
    
    print(len(chunks))
    print(chunks[0])