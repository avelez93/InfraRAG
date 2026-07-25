"""Minimal Streamlit shell: Index / Ask / Search stubs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from infrarag import __version__
from infrarag.config import load_config


def run() -> None:
    """Launch the InfraRAG Streamlit UI."""
    config = load_config()

    st.set_page_config(page_title=config.app.name, page_icon="??", layout="wide")
    st.title(config.app.name)
    st.caption(f"Local RAG scaffolding v{__version__} -- pipeline not wired yet.")

    tab_index, tab_ask, tab_search = st.tabs(["Index", "Ask", "Search"])

    with tab_index:
        st.subheader("Index directory")
        st.write(
            "Point at a folder to differentially ingest readable documents "
            "(new or changed files only)."
        )
        default_source = config.ingest.source_dir or ""
        source_dir = st.text_input(
            "Source directory",
            value=default_source,
            placeholder="/path/to/your/documents",
        )
        if st.button("Dump directory (differential)", type="primary"):
            if not source_dir.strip():
                st.warning("Set a source directory first.")
            else:
                path = Path(source_dir)
                st.info(
                    f"Ingest stub: would differentially index `{path}` "
                    f"(extensions: {', '.join(config.ingest.include_extensions)}). "
                    "Coming soon."
                )
                st.json(
                    {
                        "added": 0,
                        "updated": 0,
                        "unchanged": 0,
                        "skipped": 0,
                        "errors": [],
                        "note": "ingest_directory not implemented yet",
                    }
                )

        with st.expander("Ingest settings"):
            st.write(f"Differential: `{config.ingest.differential}`")
            st.write(f"Manifest: `{config.ingest.manifest_path}`")
            st.write(f"Chunk size / overlap: `{config.chunking.size}` / `{config.chunking.overlap}`")

    with tab_ask:
        st.subheader("Ask")
        st.write("Chat over the indexed corpus. Attach a file and/or a URL for this turn only.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        uploaded = st.file_uploader(
            "Attach document (ephemeral)",
            type=[ext.lstrip(".") for ext in config.ingest.include_extensions],
        )
        url = st.text_input("Attach web URL (ephemeral)", placeholder="https://example.com/page")

        prompt = st.chat_input("Ask a question about your documents")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            attachment_bits: list[str] = []
            if uploaded is not None:
                attachment_bits.append(f"file:{uploaded.name}")
            if url.strip():
                attachment_bits.append(f"url:{url.strip()}")
            note = (
                "Ask stub: RAG pipeline not implemented yet. "
                + (f"Attachments noted: {', '.join(attachment_bits)}" if attachment_bits else "")
            )
            st.session_state.messages.append({"role": "assistant", "content": note})
            st.rerun()

        with st.expander("RAG settings"):
            st.write(f"top_k: `{config.rag.top_k}`")
            st.write(f"temperature: `{config.rag.temperature}`")
            st.write(f"chat model: `{config.ollama.chat_model}`")

    with tab_search:
        st.subheader("Search")
        st.write("Semantic search over the corpus without generating an answer.")
        query = st.text_input("Search query", key="search_query")
        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Enter a search query.")
            else:
                st.info(
                    f"Search stub: would retrieve top_k={config.rag.top_k} chunks for "
                    f"`{query.strip()}`. Coming soon."
                )


if __name__ == "__main__":
    run()
