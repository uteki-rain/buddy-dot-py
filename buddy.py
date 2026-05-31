from __future__ import annotations
from buddy_runtime import Frame, Runtime
from backend_ollama_serve import OllamaResponder, OllamaEmbedder

# Note to self.
# Remember the concepts.
#   EVENT - ACTION - MEMORY - COGNEME - STATE
# Remember the structures.
#   INDEXER - PERIPHERY - CONTEXT - GUIDE

# Event, Action, and Memory can encode into Cogneme
# Cogneme decoding is partial
# Cogneme must be encodable into str in two ways: plain or frame title

# Sub-action cycle:
# Start with "my thoughts on what to do <colon>"
# Possible action kinds come with natural language descriptions
# Search action list via embedding relevance; e.g. "I can use /reply, /ignore"
# Each action encodes its own state machine in a compatible datatype
# States give different prompts, e.g. "recipient <colon>" or "message <colon>"
