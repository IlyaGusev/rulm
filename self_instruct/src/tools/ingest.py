import os
import glob
from typing import List

import fire
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from chromadb.config import Settings


def get_chroma_settings(output_dir):
    return Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=output_dir,
        anonymized_telemetry=False
    )


LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in LOADER_MAPPING
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    return loader.load()[0]


def load_documents(source_dir: str) -> List[Document]:
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    with ThreadPool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(all_files), desc='Loading documents') as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, all_files)):
                results.append(doc)
                pbar.update()
    return results



def ingest(
    source_dir,
    output_dir,
    embeddings_model_name,
    chunk_size: int = 200,
    chunk_overlap: int = 20
):
    # Load documents and split in chunks
    print(f"Loading documents from {source_dir}")
    documents = load_documents(source_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {source_dir}")
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} characters each)")

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=output_dir,
        client_settings=get_chroma_settings(output_dir)
    )
    db.persist()


if __name__ == "__main__":
    fire.Fire(ingest)
