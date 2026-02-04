"""Response models for chatbot streaming API."""

import json
from dataclasses import dataclass, field
from typing import Optional
import re


@dataclass
class StreamedResponse:
    """Holds the accumulated streamed response chunks."""
    user_message_id: Optional[int] = None
    assistant_message_id: Optional[int] = None
    chunks: list[dict] = field(default_factory=list)
    grouped_chunks: dict[int, list[dict]] = field(default_factory=dict)
    stopped: bool = False
    error: Optional[str] = None

    def get_by_ind(self, ind: int) -> list[dict]:
        """Get all chunks for a specific ind value."""
        return self.grouped_chunks.get(ind, [])

    def get_by_type(self, chunk_type: str, ind: Optional[int] = None) -> list[dict]:
        """Get all chunks of a specific type, optionally filtered by ind."""
        if ind is not None:
            chunks = self.get_by_ind(ind)
        else:
            chunks = self.chunks

        return [
            c["obj"] for c in chunks
            if c.get("obj", {}).get("type") == chunk_type
        ]

    def get_message(self) -> str:
        """Concatenate all message_delta chunks into final message.

        The stream contains only one message, so we concatenate across all inds.
        """
        return "".join(
            c.get("content", "") for c in self.get_by_type("message_delta")
        )

    def get_reasoning(self) -> dict[int, str]:
        """Get all reasoning texts grouped by ind.

        Returns a dict mapping ind -> concatenated reasoning text.
        """
        reasoning_by_ind = {}
        for ind, chunks in self.grouped_chunks.items():
            reasoning_chunks = [
                c["obj"] for c in chunks
                if c.get("obj", {}).get("type") == "reasoning_delta"
            ]
            if reasoning_chunks:
                reasoning_text = "".join(
                    chunk.get("reasoning", "") for chunk in reasoning_chunks
                )
                reasoning_by_ind[ind] = reasoning_text
        return reasoning_by_ind

    def get_search_tools(self) -> dict[int, dict]:
        """Get all search tool results grouped by ind.

        Returns a dict mapping ind -> search tool data with queries and documents.
        """
        search_by_ind = {}
        for ind, chunks in self.grouped_chunks.items():
            # Look for internal_search_tool_start and internal_search_tool_delta
            search_start = [
                c["obj"] for c in chunks
                if c.get("obj", {}).get("type") == "internal_search_tool_start"
            ]
            search_deltas = [
                c["obj"] for c in chunks
                if c.get("obj", {}).get("type") == "internal_search_tool_delta"
            ]

            if search_start or search_deltas:
                # Combine all queries and documents from deltas
                all_queries = []
                all_documents = []
                for delta in search_deltas:
                    all_queries.extend(delta.get("queries", []))
                    all_documents.extend(delta.get("documents", []))

                search_by_ind[ind] = {
                    "is_internet_search": search_start[0].get("is_internet_search", False) if search_start else False,
                    "queries": all_queries,
                    "documents": all_documents
                }
        return search_by_ind

    def get_final_documents(self) -> list[dict]:
        """Get final_documents from message_start chunk.

        Returns the documents that will be cited in the final message.
        """
        message_starts = self.get_by_type("message_start")
        if message_starts:
            return message_starts[0].get("final_documents", [])
        return []

    def get_citations(self) -> list[dict]:
        citations = []
        for chunk in self.chunks:
            obj = chunk.get("obj", {})
            if obj.get("type") == "citation_delta":
                citations += obj.get("citations", [])
        return citations

    def get_related_questions(self) -> list[str]:
        def parse_lines(text: str) -> list[str]:
            return [line for line in text.split('\n') if line.strip()]

        message = self.get_message()

        match = re.search(r'\[[\s\S]*?\]', message)

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return parse_lines(match.group(0))

        return parse_lines(message)

    @classmethod
    def from_jsonl_body(cls, body: str) -> "StreamedResponse":
        """Parse JSONL body into a StreamedResponse.

        Args:
            body: Raw JSONL response body from the streaming API

        Returns:
            StreamedResponse object with parsed and grouped chunks
        """
        # Parse JSONL
        chunks = []
        grouped = {}
        user_msg_id = None
        assistant_msg_id = None
        error = None
        stopped = False
        for chunk in body.strip().split('\n'):
            if not chunk:
                continue
            chunk = json.loads(chunk)
            if "user_message_id" in chunk and "reserved_assistant_message_id" in chunk:
                user_msg_id = chunk.get("user_message_id")
                assistant_msg_id = chunk.get("reserved_assistant_message_id")
                chunk = {
                    "ind": -1,
                    "obj": {
                        "type": "ids_info",
                        "user_msg_id": user_msg_id,
                        "assistant_msg_id": assistant_msg_id,
                    },
                }
            if "error" in chunk:
                error = chunk.get("error")
                chunk = {
                    "ind": -1,
                    "obj": {
                        "type": "error",
                        "error": error
                    },
                }
            chunks.append(chunk)
            if "ind" in chunk:
                ind = chunk["ind"]
                if "obj" in chunk and chunk["obj"].get("type") == "stop":
                    stopped = True
                if ind not in grouped:
                    grouped[ind] = []
                grouped[ind].append(chunk)

        return cls(
            user_message_id=user_msg_id,
            assistant_message_id=assistant_msg_id,
            chunks=chunks,
            grouped_chunks=grouped,
            stopped=stopped,
            error=error
        )
