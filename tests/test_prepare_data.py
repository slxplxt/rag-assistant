from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prepare_data import build_chunks


def test_tbank_chunking_is_unchanged():
    doc = Document(page_content="РЈСЃР»РѕРІРёСЏ С‚Р°СЂРёС„Р°. " * 500,
                   metadata={"bank": "tbank", "page": 1, "source": "fixture.pdf"})
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""], encoding_name="cl100k_base",
    )
    assert build_chunks([doc]) == splitter.split_documents([doc])
