"""Streamlit UI: Ask / Ask Deep / Search / Index wired to ingest and RAG pipelines."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from infrarag import __version__
from infrarag.config import load_config
from infrarag.ingest.pipeline import (
    ingest_directory,
    reingest_known_roots,
    resolve_known_ingest_roots,
)
from infrarag.models import ProposedFact
from infrarag.rag.attachments import resolve_attachments
from infrarag.rag.memory import propose_facts, save_confirmed_facts
from infrarag.rag.pipeline import ask_stream, search
from infrarag.rag.profiles import resolve_org_dir, resolve_user_dir
from infrarag.rag.retrieve import GatheredContext


def _init_session_state() -> None:
    defaults: dict[str, object] = {
        "messages_ask": [],
        "messages_deep": [],
        "ask_attach_bytes": None,
        "ask_attach_name": None,
        "ask_attach_url": "",
        "deep_attach_bytes": None,
        "deep_attach_name": None,
        "deep_attach_url": "",
        "pending_facts_ask": [],
        "pending_facts_deep": [],
        "pending_query_ask": "",
        "pending_query_deep": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _clear_attachments(prefix: str) -> None:
    st.session_state[f"{prefix}_attach_bytes"] = None
    st.session_state[f"{prefix}_attach_name"] = None
    st.session_state[f"{prefix}_attach_url"] = ""


def _render_composer(config, *, prefix: str) -> str | None:
    """Question bar with a + popover for document / URL attachments."""
    ext_types = [ext.lstrip(".") for ext in config.ingest.include_extensions]
    bytes_key = f"{prefix}_attach_bytes"
    name_key = f"{prefix}_attach_name"
    url_key = f"{prefix}_attach_url"

    plus_col, input_col, send_col = st.columns([0.08, 0.77, 0.15])

    with plus_col:
        with st.popover("+", help="Attach document or URL for this turn"):
            st.caption("Ephemeral attachments (this turn only)")
            choice = st.radio(
                "Attach",
                options=["Document", "URL"],
                horizontal=True,
                label_visibility="collapsed",
                key=f"{prefix}_attach_choice",
            )
            if choice == "Document":
                uploaded = st.file_uploader(
                    "Document",
                    type=ext_types,
                    key=f"{prefix}_popover_uploader",
                    label_visibility="collapsed",
                )
                if uploaded is not None:
                    st.session_state[bytes_key] = uploaded.getvalue()
                    st.session_state[name_key] = uploaded.name
                    st.session_state[url_key] = ""
                    st.caption(f"Selected: `{uploaded.name}`")
            else:
                url_val = st.text_input(
                    "URL",
                    value=st.session_state[url_key],
                    placeholder="https://example.com/page",
                    key=f"{prefix}_popover_url",
                    label_visibility="collapsed",
                )
                if st.button("Use URL", key=f"{prefix}_use_url"):
                    st.session_state[url_key] = url_val.strip()
                    st.session_state[bytes_key] = None
                    st.session_state[name_key] = None
                if st.session_state[url_key]:
                    st.caption(f"URL: `{st.session_state[url_key]}`")

            if st.session_state[name_key] or st.session_state[url_key]:
                if st.button("Clear attachment", key=f"{prefix}_clear_attach"):
                    _clear_attachments(prefix)
                    st.rerun()

    with input_col:
        qkey = f"{prefix}_question_input"
        clear_flag = f"{prefix}_clear_question"
        # Clear must happen BEFORE the widget is instantiated (Streamlit rule).
        if st.session_state.pop(clear_flag, False):
            st.session_state[qkey] = ""
        prompt = st.text_input(
            "Question",
            placeholder="Ask a question about your documents",
            label_visibility="collapsed",
            key=qkey,
        )

    with send_col:
        sent = st.button(
            "Send",
            type="primary",
            use_container_width=True,
            key=f"{prefix}_send",
        )

    bits: list[str] = []
    if st.session_state[name_key]:
        bits.append(f"file: `{st.session_state[name_key]}`")
    if st.session_state[url_key]:
        bits.append(f"url: `{st.session_state[url_key]}`")
    if bits:
        st.caption("Attached — " + " · ".join(bits))

    if sent and prompt and prompt.strip():
        return prompt.strip()
    return None


def _resolve_turn_attachments(config, *, prefix: str):
    files: list[Path] = []
    bytes_key = f"{prefix}_attach_bytes"
    name_key = f"{prefix}_attach_name"
    url_key = f"{prefix}_attach_url"
    if st.session_state[bytes_key] and st.session_state[name_key]:
        upload_dir = Path(config.app.data_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / st.session_state[name_key]
        temp_path.write_bytes(st.session_state[bytes_key])
        files.append(temp_path)
    urls = [st.session_state[url_key]] if st.session_state[url_key] else []
    if files or urls:
        return resolve_attachments(files, urls, config)
    return None


def _show_sources(
    *,
    local: list[str] | None = None,
    web: list[str] | None = None,
    gathered: GatheredContext | None = None,
) -> None:
    """Show citations in a collapsed expander."""
    if gathered is not None:
        local = list(gathered.local_citations)
        web = list(gathered.web_citations)
        profile_paths = [*gathered.profile.org_sources, *gathered.profile.user_sources]
        local = _unique([*(local or []), *profile_paths])
        web = _unique(list(web or []))
    else:
        local = _unique(list(local or []))
        web = _unique(list(web or []))
    if not local and not web:
        return
    with st.expander("Sources", expanded=False):
        if local:
            st.caption("Local / profile")
            for cite in local:
                st.markdown(f"- `{cite}`")
        if web:
            st.caption("Web")
            for cite in web:
                st.markdown(f"- {cite}")


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _sources_from_gathered(gathered: GatheredContext) -> dict[str, list[str]]:
    profile_paths = [*gathered.profile.org_sources, *gathered.profile.user_sources]
    return {
        "local": _unique([*profile_paths, *gathered.local_citations]),
        "web": _unique(list(gathered.web_citations)),
    }


def _render_memory_prompt(config, *, tab_key: str) -> None:
    facts_key = f"pending_facts_{tab_key}"
    raw_facts = st.session_state.get(facts_key) or []
    if not raw_facts:
        return
    facts = [
        ProposedFact(summary=f["summary"], text=f["text"])
        if isinstance(f, dict)
        else f
        for f in raw_facts
    ]
    with st.expander("Save to your profile memory?", expanded=True):
        st.caption("Confirm durable facts to store under your user profile memory.")
        selected: list[ProposedFact] = []
        for i, fact in enumerate(facts):
            checked = st.checkbox(
                f"{fact.summary}",
                value=True,
                key=f"{tab_key}_fact_cb_{i}",
            )
            st.caption(fact.text)
            if checked:
                selected.append(fact)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Confirm", type="primary", key=f"{tab_key}_memory_confirm"):
                if selected:
                    path, report = save_confirmed_facts(selected, config)
                    if path is None:
                        st.info("Nothing new to save (already in profile or memory).")
                    else:
                        msg = f"Saved `{path}`."
                        if report is not None:
                            msg += (
                                f" Ingest: +{report.added} ~{report.updated} "
                                f"={report.unchanged}."
                            )
                        st.success(msg)
                st.session_state[facts_key] = []
                st.rerun()
        with col_b:
            if st.button("Dismiss", key=f"{tab_key}_memory_dismiss"):
                st.session_state[facts_key] = []
                st.rerun()


def _run_ask_tab(config, *, tab_key: str, mode: str, title: str, subtitle: str) -> None:
    messages_key = f"messages_{tab_key}"
    facts_key = f"pending_facts_{tab_key}"
    query_key = f"pending_query_{tab_key}"

    st.subheader(title)
    st.write(subtitle)

    for message in st.session_state[messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            sources = message.get("sources")
            if isinstance(sources, dict):
                _show_sources(
                    local=list(sources.get("local") or []),
                    web=list(sources.get("web") or []),
                )

    _render_memory_prompt(config, tab_key=tab_key)

    prompt = _render_composer(config, prefix=tab_key)
    if prompt:
        st.session_state[messages_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            attachment_ctx = _resolve_turn_attachments(config, prefix=tab_key)
            with st.chat_message("assistant"):
                stream, gathered = ask_stream(
                    prompt, config, attachment_ctx, mode=mode
                )
                answer = st.write_stream(stream)
                _show_sources(gathered=gathered)
                if gathered.web_used and gathered.web_hits:
                    st.caption(
                        f"Web complement used "
                        f"(max local score={gathered.max_local_score:.3f})."
                    )
                elif mode == "ask" and gathered.max_local_score > config.web.score_threshold:
                    st.caption(
                        f"Local corpus sufficient "
                        f"(max score={gathered.max_local_score:.3f}); web skipped."
                    )
                full = answer if isinstance(answer, str) else "".join(answer or [])
                full = full.strip()
                st.session_state[messages_key].append(
                    {
                        "role": "assistant",
                        "content": full,
                        "sources": _sources_from_gathered(gathered),
                    }
                )

            proposals: list[ProposedFact] = []
            if config.memory.enabled and config.memory.propose_after_answer:
                with st.spinner("Checking for facts to save ..."):
                    proposals = propose_facts(
                        query=prompt,
                        answer=full,
                        config=config,
                    )
            st.session_state[facts_key] = [
                {"summary": f.summary, "text": f.text} for f in proposals
            ]
            st.session_state[query_key] = prompt
            _clear_attachments(tab_key)
            # Defer clearing the text_input until the next run, before the widget exists.
            st.session_state[f"{tab_key}_clear_question"] = True
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            err = f"Error: {exc}"
            st.error(err)
            st.session_state[messages_key].append({"role": "assistant", "content": err})


def run() -> None:
    """Launch the InfraRAG Streamlit UI."""
    config = load_config()
    _init_session_state()

    st.set_page_config(page_title=config.app.name, page_icon="📚", layout="wide")
    st.title(config.app.name)
    st.caption(f"Local RAG v{__version__}")

    with st.sidebar:
        st.header("Models")
        st.write(f"Chat: `{config.ollama.chat_model}`")
        st.write(f"Embed: `{config.ollama.embed_model}`")
        st.write(f"top_k: `{config.rag.top_k}`")
        st.write(f"Ollama: `{config.ollama.base_url}`")
        st.header("Profiles")
        st.write(f"User: `{config.profiles.user_id}`")
        st.caption(f"Org: `{resolve_org_dir(config)}`")
        st.caption(f"User dir: `{resolve_user_dir(config)}`")
        st.write(f"Web: `{'on' if config.web.enabled else 'off'}` "
                 f"(threshold={config.web.score_threshold})")

    tab_ask, tab_deep, tab_search, tab_index = st.tabs(
        ["Ask", "Ask Deep", "Search", "Index"]
    )

    with tab_ask:
        _run_ask_tab(
            config,
            tab_key="ask",
            mode="ask",
            title="Ask",
            subtitle=(
                "Chat over the indexed corpus (and profiles). "
                "Web search complements only when local max score ≤ "
                f"{config.web.score_threshold}. "
                "Use **+** to attach a document or URL for this turn."
            ),
        )

    with tab_deep:
        st.caption("Always searches local corpus and the web (slower).")
        _run_ask_tab(
            config,
            tab_key="deep",
            mode="deep",
            title="Ask Deep",
            subtitle=(
                "Always retrieves local corpus and the web, then answers. "
                "Slower than Ask. Use **+** for ephemeral attachments."
            ),
        )

    with tab_search:
        st.subheader("Search")
        st.write("Semantic search over the corpus without generating an answer.")
        query = st.text_input("Search query", key="search_query")
        if st.button("Search", type="primary"):
            if not query.strip():
                st.warning("Enter a search query.")
            else:
                with st.spinner("Searching ..."):
                    try:
                        hits = search(query.strip(), config)
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))
                    else:
                        if not hits:
                            st.info("No hits.")
                        for hit in hits:
                            st.markdown(
                                f"**`{hit.chunk.source_path}`** (score={hit.score:.3f})"
                            )
                            st.text(
                                hit.chunk.text[:500]
                                + ("..." if len(hit.chunk.text) > 500 else "")
                            )

    with tab_index:
        st.subheader("Index directory")
        st.write(
            "Point at a folder to differentially ingest readable documents "
            "(new or changed files only). You can also index "
            "`profiles/org` or `profiles/users/<id>` so profile and memory "
            "files become searchable."
        )
        default_source = config.ingest.source_dir or ""
        source_dir = st.text_input(
            "Source directory",
            value=default_source,
            placeholder="/path/to/your/documents",
        )
        st.caption(
            f"Shortcuts — org: `{resolve_org_dir(config)}` · "
            f"user: `{resolve_user_dir(config)}`"
        )

        def _run_ingest(path: Path, *, force: bool = False) -> None:
            with st.spinner(f"{'Re-indexing' if force else 'Indexing'} {path} ..."):
                try:
                    report = ingest_directory(path, config, force=force)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                else:
                    st.success("Ingest finished.")
                    st.json(
                        {
                            "added": report.added,
                            "updated": report.updated,
                            "unchanged": report.unchanged,
                            "skipped": report.skipped,
                            "errors": report.errors,
                        }
                    )

        known_roots = resolve_known_ingest_roots(config)
        if known_roots:
            st.caption("Known indexed directories:")
            for root in known_roots:
                st.caption(f"- `{root}`")
        else:
            st.caption(
                "No known indexed directories yet. Index a folder once; "
                "it will be remembered for Re-index all."
            )

        col_dump, col_org, col_user, col_re = st.columns(4)
        with col_dump:
            if st.button("Dump directory (differential)", type="primary"):
                if not str(source_dir).strip():
                    st.warning("Set a source directory first.")
                else:
                    _run_ingest(Path(str(source_dir)))
        with col_org:
            if st.button("Index org profile"):
                _run_ingest(resolve_org_dir(config))
        with col_user:
            if st.button("Index user profile"):
                _run_ingest(resolve_user_dir(config))
        with col_re:
            if st.button("Re-index all", help="Force re-embed every known indexed directory"):
                if not known_roots:
                    st.warning("Nothing to re-index yet. Index a directory first.")
                else:
                    with st.spinner(f"Re-indexing {len(known_roots)} director(y/ies) ..."):
                        try:
                            results = reingest_known_roots(config)
                        except Exception as exc:  # noqa: BLE001
                            st.error(str(exc))
                        else:
                            st.success(f"Re-index finished ({len(results)} root(s)).")
                            for root, report in results:
                                st.markdown(f"**`{root}`**")
                                st.json(
                                    {
                                        "added": report.added,
                                        "updated": report.updated,
                                        "unchanged": report.unchanged,
                                        "skipped": report.skipped,
                                        "errors": report.errors,
                                    }
                                )

        with st.expander("Ingest settings"):
            st.write(f"Differential: `{config.ingest.differential}`")
            st.write(f"Manifest: `{config.ingest.manifest_path}`")
            st.write(
                f"Chunk size / overlap: `{config.chunking.size}` / `{config.chunking.overlap}`"
            )
            st.write(f"Extensions: {', '.join(config.ingest.include_extensions)}")


if __name__ == "__main__":
    run()
