from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from .utils import normalize_path_text, prefix_matches_path, sha256_text

DEFAULT_COLLECTIONS_PATH = Path("data_index/collections.yml")
DEFAULT_COLLECTION_NAME = "default"
DEFAULT_COLLECTION_MASK = "*.md"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def _normalize_rel_path(value: str | Path) -> str:
    text = normalize_path_text(value)
    return text.lstrip("/")


def _split_collection_uri(uri: str) -> tuple[str | None, str]:
    raw = uri.strip()
    if not raw:
        return None, ""
    if raw == "/":
        return None, ""
    if raw.startswith("collection://"):
        body = raw.removeprefix("collection://")
        if not body:
            return None, ""
        parts = body.split("/", 1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1].strip("/")
    return raw, ""


@dataclass(slots=True)
class ContextEntry:
    context_id: str
    collection_id: str | None
    path_prefix: str
    text: str
    created_at: str
    updated_at: str

    def matches(self, collection_id: str, relative_path: str) -> bool:
        if self.collection_id is not None and self.collection_id != collection_id:
            return False
        return prefix_matches_path(relative_path, self.path_prefix)

    def specificity(self) -> tuple[int, int]:
        scope_score = 0 if self.collection_id is None else 1
        return (len(_normalize_rel_path(self.path_prefix)), scope_score)

    def to_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "collection_id": self.collection_id,
            "path_prefix": self.path_prefix,
            "text": self.text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextEntry":
        return cls(
            context_id=str(data.get("context_id") or _new_id()),
            collection_id=(
                str(data["collection_id"])
                if data.get("collection_id") not in {None, ""}
                else None
            ),
            path_prefix=str(data.get("path_prefix") or ""),
            text=str(data.get("text") or ""),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )


@dataclass(slots=True)
class CollectionConfig:
    collection_id: str
    name: str
    root_path: str
    mask: str = DEFAULT_COLLECTION_MASK
    include_by_default: bool = True
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    def root_path_resolved(self) -> Path:
        return Path(self.root_path).expanduser().resolve()

    def to_dict(self) -> dict:
        return {
            "collection_id": self.collection_id,
            "name": self.name,
            "root_path": self.root_path,
            "mask": self.mask,
            "include_by_default": self.include_by_default,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CollectionConfig":
        return cls(
            collection_id=str(data.get("collection_id") or _new_id()),
            name=str(data.get("name") or DEFAULT_COLLECTION_NAME),
            root_path=str(data.get("root_path") or "."),
            mask=str(data.get("mask") or DEFAULT_COLLECTION_MASK),
            include_by_default=bool(data.get("include_by_default", True)),
            created_at=str(data.get("created_at") or _now_iso()),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )


@dataclass(slots=True)
class CollectionRegistry:
    path: Path
    collections: list[CollectionConfig] = field(default_factory=list)
    contexts: list[ContextEntry] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_COLLECTIONS_PATH) -> "CollectionRegistry":
        path_obj = Path(path)
        if not path_obj.exists():
            return cls(path=path_obj)

        data = yaml.safe_load(path_obj.read_text(encoding="utf-8")) or {}
        collections = [
            CollectionConfig.from_dict(item)
            for item in data.get("collections", [])
            if isinstance(item, dict)
        ]
        contexts = [
            ContextEntry.from_dict(item)
            for item in data.get("contexts", [])
            if isinstance(item, dict)
        ]
        return cls(path=path_obj, collections=collections, contexts=contexts)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "collections": [item.to_dict() for item in self.collections],
            "contexts": [item.to_dict() for item in self.contexts],
        }
        self.path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    def _touch(self) -> None:
        if self.path:
            self.save()

    def _find_index(self, identifier: str) -> int:
        needle = identifier.strip()
        if not needle:
            raise RuntimeError("Collection identifier is empty.")

        for idx, collection in enumerate(self.collections):
            if collection.collection_id == needle or collection.name == needle:
                return idx
            if normalize_path_text(collection.root_path, absolute=True) == normalize_path_text(
                needle, absolute=True
            ):
                return idx
        raise RuntimeError(f"Collection not found: {identifier}")

    def find_collection(self, identifier: str) -> CollectionConfig:
        return self.collections[self._find_index(identifier)]

    def find_by_root(self, root_path: str | Path) -> CollectionConfig | None:
        root_norm = normalize_path_text(root_path, absolute=True)
        for collection in self.collections:
            if normalize_path_text(collection.root_path, absolute=True) == root_norm:
                return collection
        return None

    def ensure_collection_for_source(
        self,
        source: str | Path,
        *,
        name: str = DEFAULT_COLLECTION_NAME,
        mask: str = DEFAULT_COLLECTION_MASK,
        include_by_default: bool = True,
    ) -> CollectionConfig:
        source_path = Path(source).expanduser().resolve()
        existing = self.find_by_root(source_path)
        if existing is not None:
            return existing

        existing_names = {item.name for item in self.collections}
        candidate_name = name
        if candidate_name in existing_names:
            base_name = source_path.name or name
            candidate_name = base_name
            suffix = 2
            while candidate_name in existing_names:
                candidate_name = f"{base_name}-{suffix}"
                suffix += 1

        collection = CollectionConfig(
            collection_id=_new_id(),
            name=candidate_name,
            root_path=str(source_path),
            mask=mask,
            include_by_default=include_by_default,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self.collections.append(collection)
        self.save()
        return collection

    def add_collection(
        self,
        *,
        name: str,
        root_path: str | Path,
        mask: str = DEFAULT_COLLECTION_MASK,
        include_by_default: bool = True,
    ) -> CollectionConfig:
        if any(item.name == name for item in self.collections):
            raise RuntimeError(f"Collection already exists: {name}")
        root_norm = Path(root_path).expanduser().resolve()
        if self.find_by_root(root_norm) is not None:
            raise RuntimeError(f"A collection already uses root path: {root_norm}")

        collection = CollectionConfig(
            collection_id=_new_id(),
            name=name,
            root_path=str(root_norm),
            mask=mask,
            include_by_default=include_by_default,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self.collections.append(collection)
        self.save()
        return collection

    def rename_collection(self, identifier: str, new_name: str) -> CollectionConfig:
        if any(item.name == new_name for item in self.collections):
            raise RuntimeError(f"Collection already exists: {new_name}")
        idx = self._find_index(identifier)
        self.collections[idx].name = new_name
        self.collections[idx].updated_at = _now_iso()
        self.save()
        return self.collections[idx]

    def remove_collection(self, identifier: str) -> CollectionConfig:
        idx = self._find_index(identifier)
        removed = self.collections.pop(idx)
        self.contexts = [
            ctx
            for ctx in self.contexts
            if ctx.collection_id is None or ctx.collection_id != removed.collection_id
        ]
        self.save()
        return removed

    def list_collections(self) -> list[CollectionConfig]:
        return sorted(self.collections, key=lambda item: item.name.lower())

    def _resolve_scope(self, target: str | None) -> str | None:
        if target is None:
            return None
        scope = target.strip()
        if not scope or scope == "/":
            return None
        if scope.startswith("collection://"):
            scope, _prefix = _split_collection_uri(scope)
            if scope is None:
                return None
        return self.find_collection(scope).collection_id

    def add_context(
        self,
        *,
        target: str | None,
        path_prefix: str,
        text: str,
    ) -> ContextEntry:
        collection_id, target_prefix = self.resolve_context_target(target)
        prefix_norm = _normalize_rel_path(path_prefix or target_prefix)
        for idx, context in enumerate(self.contexts):
            if context.collection_id == collection_id and context.path_prefix == prefix_norm:
                self.contexts[idx].text = text
                self.contexts[idx].updated_at = _now_iso()
                self.save()
                return self.contexts[idx]
        context = ContextEntry(
            context_id=_new_id(),
            collection_id=collection_id,
            path_prefix=prefix_norm,
            text=text,
            created_at=_now_iso(),
            updated_at=_now_iso(),
        )
        self.contexts.append(context)
        self.save()
        return context

    def list_contexts(self, target: str | None = None) -> list[ContextEntry]:
        if target is None:
            return sorted(self.contexts, key=lambda item: (item.collection_id or "", item.path_prefix))

        collection_id, prefix = self.resolve_context_target(target)
        if target.strip() == "/":
            matches = [ctx for ctx in self.contexts if ctx.collection_id is None]
        else:
            matches = [
                ctx
                for ctx in self.contexts
                if ctx.collection_id == collection_id
                and (not prefix or prefix_matches_path(ctx.path_prefix, prefix))
            ]
        return sorted(matches, key=lambda item: (item.collection_id or "", item.path_prefix))

    def remove_context(self, target: str | None, path_prefix: str) -> ContextEntry:
        collection_id, target_prefix = self.resolve_context_target(target)
        prefix_norm = _normalize_rel_path(path_prefix or target_prefix)
        for idx, context in enumerate(self.contexts):
            if context.collection_id == collection_id and context.path_prefix == prefix_norm:
                removed = self.contexts.pop(idx)
                self.save()
                return removed
        scope = target or "/"
        raise RuntimeError(f"Context not found for {scope}: {prefix_norm}")

    def contexts_for(self, collection_id: str, relative_path: str) -> list[ContextEntry]:
        rel_norm = _normalize_rel_path(relative_path)
        matches = [
            ctx
            for ctx in self.contexts
            if ctx.matches(collection_id, rel_norm)
        ]
        return sorted(matches, key=lambda item: item.specificity())

    def render_context_text(self, collection_id: str, relative_path: str) -> str:
        contexts = self.contexts_for(collection_id, relative_path)
        parts = [ctx.text.strip() for ctx in contexts if ctx.text.strip()]
        return "\n\n".join(parts).strip()

    def context_hash(self, collection_id: str, relative_path: str) -> str:
        return sha256_text(self.render_context_text(collection_id, relative_path))

    def collection_context_label(self, collection_id: str) -> str:
        collection = self.find_collection(collection_id)
        return collection.name

    def collection_ids(self) -> list[str]:
        return [item.collection_id for item in self.collections]

    def default_collections(self) -> list[CollectionConfig]:
        included = [item for item in self.collections if item.include_by_default]
        return included or list(self.collections)

    def collection_uri(self, collection_id: str, relative_path: str = "") -> str:
        collection = self.find_collection(collection_id)
        suffix = relative_path.strip("/")
        if suffix:
            return f"collection://{collection.name}/{suffix}"
        return f"collection://{collection.name}"

    def resolve_context_target(self, uri_or_name: str | None) -> tuple[str | None, str]:
        if uri_or_name is None:
            return None, ""
        scope, prefix = _split_collection_uri(uri_or_name)
        if scope is None:
            return None, prefix
        collection_id = self.find_collection(scope).collection_id
        return collection_id, prefix
