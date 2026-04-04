from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import ChunkDraft, DocumentRecord
from .tokenize import rough_token_count, split_long_text_by_tokens
from .utils import sha256_text

_TAG_RE = re.compile(r"(?<!\w)#[\w\-\u4e00-\u9fff]+")
_LINK_RE = re.compile(r"(?:https?://\S+|zotero://\S+|file:///\S+)")
_WIKI_RE = re.compile(r"\[\[([^\]]+)\]\]")
_CODE_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<code>.*?)```", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass(slots=True)
class Section:
    section_path: str
    text: str


def _extract_title(raw: str, default: str) -> str:
    match = _H1_RE.search(raw)
    if not match:
        return default
    return match.group(1).strip()


def _strip_tags_from_text(text: str) -> str:
    lines = [_TAG_RE.sub("", line) for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\s+([,.;:!?)}\]])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()


def _split_sections(raw: str, title: str) -> list[Section]:
    lines = raw.splitlines()
    sections: list[Section] = []

    current_heading = title
    buffer: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if buffer:
                sections.append(Section(current_heading, "\n".join(buffer).strip()))
                buffer = []
            current_heading = line[3:].strip() or title
            continue
        if line.startswith("# "):
            continue
        buffer.append(line)

    if buffer:
        sections.append(Section(current_heading, "\n".join(buffer).strip()))

    if not sections:
        sections.append(Section(title, raw.strip()))

    return [s for s in sections if s.text.strip()]


def _extract_code_and_text(section_text: str) -> tuple[str, list[tuple[str, str]]]:
    code_chunks: list[tuple[str, str]] = []
    cursor = 0
    prose_parts: list[str] = []

    for match in _CODE_BLOCK_RE.finditer(section_text):
        start, end = match.span()
        before = _strip_tags_from_text(section_text[cursor:start])
        prose_parts.append(before)

        paragraph_candidates = [p.strip() for p in re.split(r"\n\s*\n", before) if p.strip()]
        context_prefix = paragraph_candidates[-1] if paragraph_candidates else ""

        language = match.group("lang").strip()
        code = _strip_tags_from_text(match.group("code").strip())
        if code:
            code_text = f"language: {language}\n{code}" if language else code
            code_chunks.append((code_text, context_prefix))
        cursor = end

    prose_parts.append(_strip_tags_from_text(section_text[cursor:]))
    prose_text = "\n".join(p for p in prose_parts if p).strip()
    return prose_text, code_chunks


def _join_context_parts(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(cleaned)


def parse_markdown(
    path: Path,
    *,
    collection_id: str,
    collection_name: str,
    relative_path: str,
    context_text: str,
) -> tuple[DocumentRecord, list[ChunkDraft]]:
    raw = path.read_text(encoding="utf-8")
    title = _extract_title(raw, path.stem)

    updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    tags = sorted(set(_TAG_RE.findall(raw)))

    out_links = set(_LINK_RE.findall(raw))
    out_links.update(f"wiki://{name.strip()}" for name in _WIKI_RE.findall(raw))
    sorted_links = sorted(out_links)

    doc_id = f"{collection_id}::{relative_path}"
    source_hash = sha256_text(raw)
    context_hash = sha256_text(context_text)
    document_hash = sha256_text(f"{source_hash}::{context_hash}")
    char_count = len(raw)

    document = DocumentRecord(
        doc_id=doc_id,
        collection_id=collection_id,
        collection_name=collection_name,
        title=title,
        source_path=str(path),
        relative_path=relative_path,
        tags=tags,
        out_links=sorted_links,
        updated_at=updated_at,
        source_hash=source_hash,
        context_hash=context_hash,
        document_hash=document_hash,
        char_count=char_count,
    )

    sections = _split_sections(raw, title)

    text_chunks: list[tuple[str, str]] = []
    code_chunks: list[tuple[str, str, str]] = []

    for sec in sections:
        prose, code_list = _extract_code_and_text(sec.text)
        if prose.strip():
            text_chunks.append((sec.section_path, prose.strip()))
        for code_text, context_prefix in code_list:
            code_chunks.append((sec.section_path, code_text, context_prefix))

    drafts: list[ChunkDraft] = []

    def append_text_chunk(section_path: str, text: str, idx: int) -> None:
        clean = text.strip()
        if not clean:
            return
        search_text = _join_context_parts(context_text, clean)
        token_count = rough_token_count(search_text)
        chunk_id = f"{doc_id}::text::{idx}"
        drafts.append(
            ChunkDraft(
                chunk_id=chunk_id,
                doc_id=doc_id,
                collection_id=collection_id,
                collection_name=collection_name,
                title=title,
                source_path=str(path),
                relative_path=relative_path,
                section_path=section_path,
                chunk_type="text",
                context_prefix="",
                context_text=context_text,
                text=clean,
                search_text=search_text,
                token_count=token_count,
                embedding_hash=sha256_text(f"text::{search_text}"),
                tags=tags,
                out_links=sorted_links,
                updated_at=updated_at,
            )
        )

    text_chunk_idx = 0
    if char_count < 1200:
        merged = "\n\n".join(text for _, text in text_chunks).strip()
        append_text_chunk(title, merged, text_chunk_idx)
        text_chunk_idx += 1
    elif char_count <= 3500:
        for section_path, text in text_chunks:
            append_text_chunk(section_path, text, text_chunk_idx)
            text_chunk_idx += 1
    else:
        for section_path, text in text_chunks:
            parts = split_long_text_by_tokens(text, window=800, overlap=120)
            if not parts:
                continue
            for part_idx, part in enumerate(parts):
                append_text_chunk(f"{section_path}::part{part_idx + 1}", part, text_chunk_idx)
                text_chunk_idx += 1

    code_idx = 0
    for section_path, code_text, context_prefix in code_chunks:
        searchable = _join_context_parts(
            context_text,
            f"context:\n{context_prefix}" if context_prefix else "",
            f"code:\n{code_text}",
        )
        token_count = rough_token_count(searchable)
        chunk_id = f"{doc_id}::code::{code_idx}"
        drafts.append(
            ChunkDraft(
                chunk_id=chunk_id,
                doc_id=doc_id,
                collection_id=collection_id,
                collection_name=collection_name,
                title=title,
                source_path=str(path),
                relative_path=relative_path,
                section_path=section_path,
                chunk_type="code",
                context_prefix=context_prefix,
                context_text=context_text,
                text=code_text,
                search_text=searchable,
                token_count=token_count,
                embedding_hash=sha256_text(f"code::{searchable}"),
                tags=tags,
                out_links=sorted_links,
                updated_at=updated_at,
            )
        )
        code_idx += 1

    return document, drafts
