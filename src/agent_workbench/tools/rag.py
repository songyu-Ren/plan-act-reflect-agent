from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_workbench.memory.long_vector import VectorMemory
from agent_workbench.settings import Settings


class RAGTool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.vector_memory = VectorMemory(settings)
        self.corpus_path = Path("data/corpus")
    
    def search(self, query: str, k: Optional[int] = None) -> Dict[str, Any]:
        """Search the vector memory for relevant documents"""
        try:
            if k is None:
                k = self.settings.retrieval.k
            
            results = self.vector_memory.search(query, k)
            
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Search failed: {str(e)}",
                "query": query,
                "results": [],
                "count": 0
            }
    
    def ingest_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest documents into vector memory"""
        try:
            doc_ids = self.vector_memory.add_documents(documents)
            
            return {
                "success": True,
                "ingested_count": len(doc_ids),
                "doc_ids": doc_ids
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Ingestion failed: {str(e)}",
                "ingested_count": 0,
                "doc_ids": []
            }
    
    def ingest_corpus(self) -> Dict[str, Any]:
        """Ingest all documents from the corpus directory"""
        try:
            if not self.corpus_path.exists():
                return {
                    "success": False,
                    "error": f"Corpus directory not found: {self.corpus_path}",
                    "ingested_count": 0,
                    "doc_ids": []
                }
            
            documents = []
            
            # Process text files
            for text_file in self.corpus_path.glob("*.txt"):
                try:
                    content = text_file.read_text(encoding='utf-8')
                    documents.append({
                        "id": f"txt_{text_file.stem}",
                        "text": content,
                        "metadata": {
                            "source": str(text_file),
                            "type": "text",
                            "filename": text_file.name
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not read {text_file}: {e}")
            
            # Process markdown files
            for md_file in self.corpus_path.glob("*.md"):
                try:
                    content = md_file.read_text(encoding='utf-8')
                    documents.append({
                        "id": f"md_{md_file.stem}",
                        "text": content,
                        "metadata": {
                            "source": str(md_file),
                            "type": "markdown",
                            "filename": md_file.name
                        }
                    })
                except Exception as e:
                    print(f"Warning: Could not read {md_file}: {e}")
            
            # Process PDF files (basic text extraction)
            try:
                import PyPDF2
                for pdf_file in self.corpus_path.glob("*.pdf"):
                    try:
                        with open(pdf_file, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            content = ""
                            for page in reader.pages:
                                content += page.extract_text() + "\n"
                            
                            documents.append({
                                "id": f"pdf_{pdf_file.stem}",
                                "text": content.strip(),
                                "metadata": {
                                    "source": str(pdf_file),
                                    "type": "pdf",
                                    "filename": pdf_file.name,
                                    "pages": len(reader.pages)
                                }
                            })
                    except Exception as e:
                        print(f"Warning: Could not read {pdf_file}: {e}")
            except ImportError:
                print("PyPDF2 not available, skipping PDF files")
            
            if not documents:
                return {
                    "success": False,
                    "error": "No documents found to ingest",
                    "ingested_count": 0,
                    "doc_ids": []
                }
            
            return self.ingest_documents(documents)
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Corpus ingestion failed: {str(e)}",
                "ingested_count": 0,
                "doc_ids": []
            }
    
    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Get a specific document by ID"""
        try:
            doc = self.vector_memory.get_document(doc_id)
            
            if doc:
                return {
                    "success": True,
                    "document": doc
                }
            else:
                return {
                    "success": False,
                    "error": f"Document not found: {doc_id}",
                    "document": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get document: {str(e)}",
                "document": None
            }
    
    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document by ID"""
        try:
            success = self.vector_memory.delete_document(doc_id)
            
            return {
                "success": success,
                "deleted": success,
                "doc_id": doc_id
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete document: {str(e)}",
                "deleted": False,
                "doc_id": doc_id
            }
    
    def clear_all(self) -> Dict[str, Any]:
        """Clear all documents from vector memory"""
        try:
            self.vector_memory.clear()
            
            return {
                "success": True,
                "cleared": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to clear documents: {str(e)}",
                "cleared": False
            }