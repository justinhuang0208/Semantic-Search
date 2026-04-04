from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from semsearch.markdown_ingest import parse_markdown


class MarkdownIngestTests(unittest.TestCase):
    def test_tags_stay_in_metadata_and_do_not_enter_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            md_path = tmp_path / "note.md"
            md_path.write_text(
                "# Title\n"
                "Intro about the module #verilog.\n\n"
                "## Code\n"
                "```verilog\n"
                "assign a = b;\n"
                "```\n",
                encoding="utf-8",
            )

            document, chunks = parse_markdown(
                md_path,
                collection_id="notes",
                collection_name="Notes",
                relative_path="note.md",
                context_text="",
            )

        self.assertEqual(document.tags, ["#verilog"])
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(any(chunk.chunk_type == "text" for chunk in chunks))
        self.assertTrue(any(chunk.chunk_type == "code" for chunk in chunks))

        for chunk in chunks:
            self.assertNotIn("#verilog", chunk.text)
            self.assertNotIn("#verilog", chunk.search_text)

        code_chunks = [chunk for chunk in chunks if chunk.chunk_type == "code"]
        self.assertTrue(any("assign a = b;" in chunk.text for chunk in code_chunks))


if __name__ == "__main__":
    unittest.main()
