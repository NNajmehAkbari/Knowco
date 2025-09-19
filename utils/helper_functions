import os
import sys
import vertexai
import subprocess
import numpy as np
import fitz
import pandas as pd
import unicodedata
import re

from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel, Part as PreviewPart
from vertexai.vision_models import Image as VisionImage
from vertexai.preview.generative_models import GenerativeModel, Part as PreviewPart
from sklearn.metrics.pairwise import cosine_similarity

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from chromadb.utils import embedding_functions
from typing import Any, Dict, List, Literal, Optional,  List, Tuple, Set, Union

from chunking_evaluation.utils import openai_token_count
from chunking_evaluation.chunking import ClusterSemanticChunker, FixedTokenChunker

from utils.key import json_path, project_id, location, openai_key

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
vertexai.init(project=project_id, location=location)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")


"""*********************  PDFs manager  *********************"""


def extract_tables_images_sources(
        pdf_path: str,
        image_save_dir: str,
        image_prompt: str,
        gen_model,  # e.g. your multimodal_model_15 instance
) -> List[Dict[str, Any]]:
    """
    For each page in the PDF, extract:
      - plain text
      - flattened tables
      - image explanations
    Returns a list of dicts:
      [
        {
          "document": <basename>,
          "page": <page_number>,
          "text": <string or "">,
          "tables": [<flattened table strings>],
          "images": [<image description strings>]
        },
        ...
      ]
    """
    results = []
    doc = fitz.open(pdf_path)
    base_name = os.path.basename(pdf_path).rsplit(".", 1)[0]

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        page = doc[page_idx]

        # container for this page
        page_entry: Dict[str, Any] = {
            "document": base_name,
            "page": page_num,
            "text": "",
            "tables": [],
            "images": []
        }

        # 1) Plain text
        txt = page.get_text("text").strip()
        if txt:
            page_entry["text"] = txt

        # 2) Tables (try lattice then stream)
        # tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="lattice")
        # if not tables:
        #     tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="stream")
        # for tbl in tables:
        #     # flatten_table should take a DataFrame and return a string
        #     page_entry["tables"].append(flatten_table(tbl.df))

        # 3) Images + explanation
        for img_i, img_info in enumerate(page.get_images(), start=1):
            gen_image, _ = get_image_for_gemini(
                doc, img_info, img_i, image_save_dir, base_name, page_num
            )
            desc = explain_image(gen_model, gen_image, image_prompt)
            page_entry["images"].append(desc)

        results.append(page_entry)

    return results


def process_pdfs(
    pdf_folder_path: str,
    image_save_dir: str,
    image_prompt: str,
    gen_model: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process all PDF files in the given folder using the `extract_with_tables_and_images_with_sources`
    function. Saves images into `image_save_dir` and uses the given prompt and generative model.

    Args:
        pdf_folder_path (str): Path to the folder containing PDF files.
        image_save_dir (str): Directory where extracted images will be stored.
        image_prompt (str): Prompt for image interpretation.
        gen_model (Any): Generative model instance for text/image interpretation.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A mapping from PDF filename to page-level extracted content.

    Raises:
        FileNotFoundError: If the PDF folder does not exist.
        RuntimeError: If extraction of a specific PDF fails.
    """
    if not os.path.isdir(pdf_folder_path):
        raise FileNotFoundError(f"PDF folder not found: {pdf_folder_path}")

    os.makedirs(image_save_dir, exist_ok=True)
    results: Dict[str, List[Dict[str, Any]]] = {}

    for f_name in sorted(os.listdir(pdf_folder_path)):
        if not f_name.lower().endswith(".pdf"):
            logger.debug(f"Skipping non-PDF file: {f_name}")
            continue

        pdf_path = os.path.join(pdf_folder_path, f_name)
        try:
            page_entries = extract_tables_images_sources(
                pdf_path=pdf_path,
                image_save_dir=image_save_dir,
                image_prompt=image_prompt,
                gen_model=gen_model,
            )

            logger.info(f"Processed {f_name} with {len(page_entries)} pages.")
            for entry in page_entries:
                logger.debug(
                    f"Page {entry['page']} - Text length: {len(entry['text'] or '')}, "
                    f"Tables: {len(entry['tables'])}, Images: {len(entry['images'])}"
                )

            results[f_name] = page_entries

        except Exception as e:
            logger.error(f"Failed to process {f_name}: {e}", exc_info=True)
            # optionally continue processing next file instead of raising:
            raise RuntimeError(f"Error processing {f_name}") from e

    return results


def process_pdf_folder(
    pdf_folder_path: str,
    image_save_dir: str,
    image_prompt: str,
    gen_model: Any,
) -> Dict[str, Any]:
    """
    Process all PDF files in `pdf_folder_path` with `extract_with_tables_and_images`,
    saving images into `image_save_dir` and using `image_prompt` & `gen_model` for generation.

    Args:
        pdf_folder_path: path to the directory containing .pdf files
        image_save_dir:   directory where extracted images will be stored
        image_prompt:     prompt to use when generating image descriptions
        gen_model:        your GenAI model instance

    Returns:
        A dict mapping each PDF filename to the extractor’s output.
    """
    results = {}

    for f_name in sorted(os.listdir(pdf_folder_path)):
        if not f_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_folder_path, f_name)
        output = extract_tables_images_sources(
            pdf_path=pdf_path,
            image_save_dir=image_save_dir,
            image_prompt=image_prompt,
            gen_model=gen_model,
        )
        results[f_name] = output

    return results


"""*********************  Chunk Manager  *********************"""


def chunk_text(document: str, chunk_size: int, overlap: int=100) -> List[str]:
    chunks = []
    stride = chunk_size - overlap
    for start in range(0, len(document), stride):
        chunks.append(document[start : start + chunk_size])
    return chunks


def chunk_document(
    document: str,
    method: str,
    **kwargs
) -> List[Union[str, object]]:
    """
    Split `document` into chunks according to `method`.

    Args:
      document: the full text to chunk.
      method: one of
        - "basic"             → simple char chunks
        - "fixed_token"       → FixedTokenChunker
        - "cluster"           → ClusterSemanticChunker
      **kwargs: parameters specific to each method.

    Returns:
      List of chunk strings (or Document objects for semantic_lc).
    """
    method = method.lower()

    if method == "basic":
        size = kwargs.get("chunk_size", 400)
        overlap = kwargs.get("overlap", 200)
        return chunk_text(document, size, overlap)

    if method == "fixed_token":
        chunker = FixedTokenChunker(
            chunk_size=kwargs.get("chunk_size", 400),
            chunk_overlap=kwargs.get("chunk_overlap", 200),
            encoding_name=kwargs.get("encoding_name", "cl100k_base"),
        )
        return chunker.split_text(document)

    if method == "cluster":
        embed_fn = kwargs["embedding_function"]
        chunker = ClusterSemanticChunker(
            embedding_function=embed_fn,
            max_chunk_size=kwargs.get("max_chunk_size", 400),
            length_function=kwargs.get("length_function", openai_token_count)
        )
        return chunker.split_text(document)


"""*********************  Embedding Manager  *********************"""


def embed_chunks(chunks, embedding_model):
    """
    Safely embed a list of chunks (which may be objects with a .text attribute or plain strings),
    one at a time to avoid token overflow.

    Args:
        chunks: Iterable of chunk objects or strings.
        embedding_model: A pretrained TextEmbeddingModel instance.

    Returns:
        chunk_vectors: List[np.ndarray] of successfully embedded chunk vectors.
        failed_indices: List[int] of indices for chunks that failed to embed.
    """
    # Ensure each chunk is a string
    safe_texts = [c.text if hasattr(c, "text") else str(c) for c in chunks]

    chunk_vectors = []
    failed_indices = []

    for i, txt in enumerate(safe_texts):
        try:
            # Embed one chunk at a time
            embedding = embedding_model.get_embeddings([txt])[0]
            vec = np.array(embedding.values, dtype=np.float32)
            chunk_vectors.append(vec)
        except Exception as e:
            print(f"❌ Failed embedding chunk {i} (len={len(txt)}): {e}")
            failed_indices.append(i)

    return safe_texts, chunk_vectors, failed_indices


"""*********************  Prompts  *********************"""


def prompts_call(prompt_name: str, prompt_path: str = "/Users/najmehakbari/KnowCo/content/promptknowco.csv") -> str:
    """
    Return a stored prompt template by name.
    Args:
        prompt_name (str): Name of the prompt to retrieve (e.g., 'image_description_prompt').
        prompt_path (str, optional): CSV file path containing Prompt_Name and Prompt columns.
    Returns:
        str: The prompt template text.
    Raises:
        ValueError: If no prompt is found for the given name.
    """
    df = pd.read_csv(prompt_path)
    result = df[df["Prompt_Name"] == prompt_name]["Prompt"]
    if result.empty:
        raise ValueError(f"No prompt found for {prompt_name}")
    return result.values[0]


image_description_prompt = prompts_call("image_description_prompt")
answer_query_prompt = prompts_call("answer_query")

"""*********************  Retrieve  *********************"""


def retrieve_top_k(query: str, embedding_model, texts, vectors, k=3):
    # Embed the query
    query_vec = embedding_model.get_embeddings([query])[0].values
    # Compute cosine similarity
    sims = cosine_similarity([query_vec], vectors)[0]
    # Rank by similarity
    top_indices = np.argsort(sims)[::-1][:k]
    return [(texts[i], sims[i]) for i in top_indices]


def embedding_functions(texts: list[str]) -> list[list[float]]:
    response = embedding_model.get_embeddings(texts)
    return [emb.values for emb in response]


"""*********************  Other  *********************"""


def openai_token_count(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text, disallowed_special=()))


def count_tokens(text, model="cl100k_base"):
    """Count tokens in a text string using tiktoken"""
    encoder = tiktoken.get_encoding(model)
    return print(f"Number of tokens: {len(encoder.encode(text))}")


def clean_excerpt(text: str, normalize_quotes: bool = True) -> str:
    # 0) Unicode normalization (ligatures, full-width → ASCII, etc.)
    text = unicodedata.normalize("NFKC", text)

    # A) Remove zero-width and BOM chars
    text = re.sub(r"[\u200B\uFEFF]", "", text)

    # 1) Strip common headers/footers:
    text = re.sub(r"^Page\s+\d+\s+of\s+\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # 2) Fix hyphen-splits across line breaks:
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # 3) Re-flow accidental line-wraps within paragraphs
    #    (only merge single newlines that don’t end in punctuation)
    text = re.sub(
        r"(?<![\.\!\?\:])\n+",
        " ",
        text
    )

    # 4) Collapse multiple newlines into exactly two (real paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5) Trim whitespace on each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # 6) Collapse runs of spaces/tabs into one space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 7) Optional: normalize fancy quotes & dashes
    if normalize_quotes:
        replace_map = {
            "“": '"', "”": '"', "‘": "'", "’": "'",
            "–": "-", "—": "-",
        }
        for fancy, simple in replace_map.items():
            text = text.replace(fancy, simple)

    # 8) Optional: strip URLs / emails / bracketed citations (leave your DP codes)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    # leave "(DP E1-7_05)" intact, so no generic parenthesis removal

    return text
