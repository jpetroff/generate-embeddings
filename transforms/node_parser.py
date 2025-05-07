from llama_index.core import Settings
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser, TokenTextSplitter


# node_parser = SentenceWindowNodeParser.from_defaults()
# node_parser = SemanticSplitterNodeParser.from_defaults(
#     embed_model=Settings.embed_model,
#     buffer_size=96,
#     include_metadata=True
#     )


node_parser = TokenTextSplitter.from_defaults(
    chunk_size=Settings.chunk_size,
    chunk_overlap=Settings.chunk_overlap,
    callback_manager=Settings.callback_manager
)