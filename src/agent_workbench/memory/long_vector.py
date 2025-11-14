from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from agent_workbench.settings import Settings


class VectorMemory:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_path = Path(settings.paths.vector_index_dir)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.model = SentenceTransformer(settings.retrieval.model_name)
        self.index_file = self.index_path / "faiss.index"
        self.mapping_file = self.index_path / "mapping.json"
        
        self.index = None
        self.mapping = {}  # index -> {text, metadata, doc_id}
        self.next_index = 0
        
        self._load_index()
    
    def _load_index(self) -> None:
        try:
            import faiss
            if self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                with open(self.mapping_file, "r") as f:
                    self.mapping = json.load(f)
                self.next_index = max([int(k) for k in self.mapping.keys()]) + 1 if self.mapping else 0
            else:
                # Initialize empty index
                dimension = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        except ImportError:
            # Fallback to simple in-memory storage if FAISS not available
            self.index = None
            self.mapping = {}
    
    def _save_index(self) -> None:
        if self.index is not None:
            try:
                import faiss
                faiss.write_index(self.index, str(self.index_file))
                with open(self.mapping_file, "w") as f:
                    json.dump(self.mapping, f, indent=2)
            except ImportError:
                pass  # In-memory only, no persistence
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to vector store"""
        doc_ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            doc_id = doc.get("id", str(uuid.uuid4()))
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            if not text:
                continue
                
            doc_ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)
        
        if not texts:
            return []
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # Normalize
        
        if self.index is not None:
            try:
                import faiss
                self.index.add(embeddings.astype(np.float32))
                
                # Update mapping
                for i, (doc_id, text, metadata) in enumerate(zip(doc_ids, texts, metadatas)):
                    self.mapping[str(self.next_index + i)] = {
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": metadata
                    }
                
                self.next_index += len(doc_ids)
                self._save_index()
            except ImportError:
                # Store in memory
                for i, (doc_id, text, metadata) in enumerate(zip(doc_ids, texts, metadatas)):
                    self.mapping[str(self.next_index + i)] = {
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": metadata,
                        "embedding": embeddings[i].tolist()
                    }
                self.next_index += len(doc_ids)
        
        return doc_ids
    
    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if k is None:
            k = self.settings.retrieval.k
        
        if not self.mapping:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        results = []
        
        if self.index is not None:
            try:
                import faiss
                scores, indices = self.index.search(query_embedding.astype(np.float32), k)
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:  # No more results
                        break
                    
                    mapping_key = str(idx)
                    if mapping_key in self.mapping:
                        doc_data = self.mapping[mapping_key]
                        results.append({
                            "doc_id": doc_data["doc_id"],
                            "text": doc_data["text"],
                            "metadata": doc_data.get("metadata", {}),
                            "score": float(score)
                        })
            except ImportError:
                # Simple cosine similarity in memory
                all_embeddings = []
                all_docs = []
                
                for key, doc_data in self.mapping.items():
                    if "embedding" in doc_data:
                        all_embeddings.append(doc_data["embedding"])
                        all_docs.append(doc_data)
                
                if all_embeddings:
                    all_embeddings = np.array(all_embeddings)
                    similarities = np.dot(all_embeddings, query_embedding[0])
                    top_k_indices = np.argsort(similarities)[-k:][::-1]
                    
                    for idx in top_k_indices:
                        doc_data = all_docs[idx]
                        results.append({
                            "doc_id": doc_data["doc_id"],
                            "text": doc_data["text"],
                            "metadata": doc_data.get("metadata", {}),
                            "score": float(similarities[idx])
                        })
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID"""
        for doc_data in self.mapping.values():
            if doc_data.get("doc_id") == doc_id:
                return {
                    "doc_id": doc_data["doc_id"],
                    "text": doc_data["text"],
                    "metadata": doc_data.get("metadata", {})
                }
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID"""
        keys_to_delete = []
        for key, doc_data in self.mapping.items():
            if doc_data.get("doc_id") == doc_id:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.mapping[key]
        
        if keys_to_delete:
            self._save_index()
            return True
        
        return False
    
    def clear(self) -> None:
        """Clear all documents"""
        self.mapping.clear()
        if self.index is not None:
            try:
                import faiss
                dimension = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatIP(dimension)
                self._save_index()
            except ImportError:
                pass
        self.next_index = 0