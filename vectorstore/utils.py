import os, hashlib
from pathlib import Path
from tqdm import tqdm
import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --- User configs ---
PDF_DIR = "./data/pdfs"                 
CHROMA_DIR = "chroma_db" # persistent folder for chromadb
COLLECTION_NAME = "papers_chunks"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 256

# Chunking params
CHUNK_WORD_TARGET = 500             
OVERLAP_WORDS = 150

# --- init ---
nltk.download('punkt')
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

client = chromadb.PersistentClient(path=CHROMA_DIR)
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"source": "research_papers"}
    )

def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode("utf-8")).hexdigest()

def file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
    
def is_reference_line(line):
    line = line.strip().lower()
    return any(keyword in line for keyword in ["references", "bibliography"])

def extract_text_from_pdf(path: str):
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"Failed to open {path}: {e}")
        return []  # skip this PDF
    pages = []
    try:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            text = page.get_text("text")
            text = text.replace("\r\n", "\n").replace("-\n", "")
            text = "\n".join([ln.strip() for ln in text.splitlines() if ln.strip() != ""])
            pages.append({"page_num": pno + 1, "text": text})
    except Exception as e:
        print(f"Error reading pages from {path}: {e}")
    finally:
        doc.close()
    return pages

def detect_section_heading(line: str):
    # simple heuristic: short lines, Title Case or ALL CAPS
    if len(line) < 200 and (line.strip().isdigit() or line.strip().isupper() or line.istitle()):
        return line.strip()
    return "Unknown"

def chunk_document_text(pages, doc_id, filename):
    # Combine all pages into one sequence
    all_text = []
    page_offsets = []
    for page in pages:
        text = page["text"]
        if not text:
            continue
        lines = text.split("\n")
        for line in lines:
            if is_reference_line(line):
                break 
            all_text.append(line)
        page_offsets.append((page["page_num"], len(all_text)))

    # Sentence tokenization
    sentences = []
    sentence_page_map = []
    for page_num, text in zip([p["page_num"] for p in pages], [p["text"] for p in pages]):
        sents = nltk.tokenize.sent_tokenize(text)
        sentences.extend(sents)
        sentence_page_map.extend([page_num] * len(sents))

    # Build chunks
    chunks = []
    cur_chunk = []
    cur_len = 0
    cur_pages = set()
    for i, sent in enumerate(sentences):
        words = nltk.word_tokenize(sent)
        cur_chunk.append(sent)
        cur_len += len(words)
        cur_pages.add(sentence_page_map[i])

        if cur_len >= CHUNK_WORD_TARGET:
            chunk_text = " ".join(cur_chunk)
            md = {
                "doc_id": doc_id,
                "source_filename": filename,
                "page_start": min(cur_pages),
                "page_end": max(cur_pages),
                "chunk_index": len(chunks),
                "word_len": cur_len,
                "sha256": sha256_text(chunk_text),
                "section_heading": detect_section_heading(cur_chunk[0]) or "Unknown"
            }
            chunks.append((chunk_text, md))

            # Prepare overlap
            if OVERLAP_WORDS > 0:
                # Take last sentences covering overlap words
                overlap = []
                overlap_len = 0
                for s in reversed(cur_chunk):
                    w = len(nltk.word_tokenize(s))
                    overlap.insert(0, s)
                    overlap_len += w
                    if overlap_len >= OVERLAP_WORDS:
                        break
                cur_chunk = overlap
                cur_len = overlap_len
                cur_pages = set([sentence_page_map[i]])
            else:
                cur_chunk = []
                cur_len = 0
                cur_pages = set()

    # Remaining chunk
    if cur_chunk:
        chunk_text = " ".join(cur_chunk)
        md = {
            "doc_id": doc_id,
            "source_filename": filename,
            "page_start": min(cur_pages),
            "page_end": max(cur_pages),
            "chunk_index": len(chunks),
            "word_len": cur_len,
            "sha256": sha256_text(chunk_text),
            "section_heading": detect_section_heading(cur_chunk[0]) if cur_chunk else None
        }
        chunks.append((chunk_text, md))

    return chunks

def ingest_pdfs(pdf_folder):
    pdf_paths = list(Path(pdf_folder).glob("*.pdf"))
    all_texts, all_metadatas, all_ids = [], [], []

    # Load existing sha256 to avoid duplicates
    existing_hashes = set()
    try:
        existing_meta = collection.get(include=["metadatas"])
        existing_hashes = {m["sha256"] for m in existing_meta["metadatas"]}
    except Exception:
        pass

    for p in tqdm(pdf_paths, desc="PDFs"):
        filename = p.name
        doc_id = file_sha256(str(p))
        pages = extract_text_from_pdf(str(p))
        chunks = chunk_document_text(pages, doc_id, filename)
        for text, md in chunks:
            if md["sha256"] in existing_hashes:
                continue
            existing_hashes.add(md["sha256"])
            all_texts.append(text)
            all_metadatas.append(md)
            chunk_id = sha256_text(doc_id + str(md["chunk_index"]))
            all_ids.append(chunk_id)

            if len(all_texts) >= BATCH_SIZE:
                embs = embedder.encode(all_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
                collection.add(documents=all_texts, embeddings=embs.tolist(), metadatas=all_metadatas, ids=all_ids)
                all_texts, all_metadatas, all_ids = [], [], []

    # Flush remainder
    if all_texts:
        embs = embedder.encode(all_texts, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        collection.add(documents=all_texts, embeddings=embs.tolist(), metadatas=all_metadatas, ids=all_ids)

    try:
        client.persist()
    except Exception:
        pass
    print("Finished ingestion.")

if __name__ == "__main__":
    ingest_pdfs(PDF_DIR)