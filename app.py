from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from openai import OpenAI
import faiss
import numpy as np

INPUT_COST = 0.00015 / 1000
OUTPUT_COST = 0.00060 / 1000

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 📄")

    client = OpenAI()

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf:
        reader = PdfReader(pdf)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        chunks = chunk_text(text)

        embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks
        )

        vectors = np.array(
            [e.embedding for e in embeddings.data],
            dtype="float32"
        )

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        question = st.text_input("Ask a question about the PDF")

        if question:
            q_embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=question
            ).data[0].embedding

            D, I = index.search(
                np.array([q_embedding], dtype="float32"),
                k=4
            )

            context = "\n\n".join([chunks[i] for i in I[0]])

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer using the provided context only."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
                ]
            )

            answer = response.choices[0].message.content
            usage = response.usage

            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

            cost = (input_tokens * INPUT_COST) + (output_tokens * OUTPUT_COST)

            st.write(response.choices[0].message.content)
            st.caption(
                f"Tokens used: {total_tokens} | "
                f"Estimated cost: ${cost:.6f}"
            )

            print("---- OpenAI Usage ----")
            print(f"Prompt tokens: {input_tokens}")
            print(f"Completion tokens: {output_tokens}")
            print(f"Total tokens: {total_tokens}")
            print(f"Estimated cost: ${cost:.6f}")

if __name__ == "__main__":
    main()
