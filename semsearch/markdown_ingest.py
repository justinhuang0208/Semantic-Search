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
        before = section_text[cursor:start]
        prose_parts.append(before)

        paragraph_candidates = [p.strip() for p in re.split(r"\n\s*\n", before) if p.strip()]
        context_prefix = paragraph_candidates[-1] if paragraph_candidates else ""

        language = match.group("lang").strip()
        code = match.group("code").strip()
        if code:
            code_text = f"language: {language}\n{code}" if language else code
            code_chunks.append((code_text, context_prefix))
        cursor = end

    prose_parts.append(section_text[cursor:])
    prose_text = "\n".join(p for p in prose_parts if p).strip()
    return prose_text, code_chunks


def parse_markdown(path: Path) -> tuple[DocumentRecord, list[ChunkDraft]]:
    raw = path.read_text(encoding="utf-8")
    title = _extract_title(raw, path.stem)

    updated_at = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    tags = sorted(set(_TAG_RE.findall(raw)))

    out_links = set(_LINK_RE.findall(raw))
    out_links.update(f"wiki://{name.strip()}" for name in _WIKI_RE.findall(raw))
    sorted_links = sorted(out_links)

    doc_id = path.stem
    doc_content_hash = sha256_text(raw)
    char_count = len(raw)

    document = DocumentRecord(
        doc_id=doc_id,
        title=title,
        source_path=str(path),
        tags=tags,
        out_links=sorted_links,
        updated_at=updated_at,
        content_hash=doc_content_hash,
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
        token_count = rough_token_count(clean)
        search_text = clean
        chunk_id = f"{doc_id}::text::{idx}"
        drafts.append(
            ChunkDraft(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                source_path=str(path),
                section_path=section_path,
                chunk_type="text",
                context_prefix="",
                text=clean,
                search_text=search_text,
                token_count=token_count,
                content_hash=sha256_text(f"text::{search_text}"),
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
        searchable = code_text if not context_prefix else f"context:\n{context_prefix}\n\ncode:\n{code_text}"
        token_count = rough_token_count(searchable)
        chunk_id = f"{doc_id}::code::{code_idx}"
        drafts.append(
            ChunkDraft(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                source_path=str(path),
                section_path=section_path,
                chunk_type="code",
                context_prefix=context_prefix,
                text=code_text,
                search_text=searchable,
                token_count=token_count,
                content_hash=sha256_text(f"code::{searchable}"),
                tags=tags,
                out_links=sorted_links,
                updated_at=updated_at,
            )
        )
        code_idx += 1

    return document, drafts
