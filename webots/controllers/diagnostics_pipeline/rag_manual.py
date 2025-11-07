"""RAG (Retrieval-Augmented Generation) system for Spot manual.

Spotマニュアルから関連情報を検索し、LLM診断のコンテキストを提供します。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("[rag] Warning: PyMuPDF not found. Install with: pip install pymupdf")

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[rag] Warning: sentence-transformers not found. Install with: pip install sentence-transformers")


class ManualRAG:
    """
    Spotマニュアルからの情報検索システム。
    
    機能:
    - PDFからテキスト抽出
    - テキストをチャンクに分割
    - 埋め込みベクトル生成
    - 類似度検索
    """
    
    def __init__(
        self,
        pdf_path: str,
        cache_dir: str = "data/manual_embeddings",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Args:
            pdf_path: Spotマニュアルのパス
            cache_dir: 埋め込みキャッシュディレクトリ
            chunk_size: テキストチャンクサイズ（文字数）
            chunk_overlap: チャンク間のオーバーラップ
            model_name: 埋め込みモデル名（多言語対応）
        """
        self.pdf_path = pdf_path
        self.cache_dir = Path(cache_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if not PYMUPDF_AVAILABLE or not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("[rag] RAG system is disabled due to missing dependencies")
            self.enabled = False
            return
        
        self.enabled = True
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 埋め込みモデル（軽量版、Jetson対応）
        print(f"[rag] Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # マニュアルを処理
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load_or_build_index()
    
    def _load_or_build_index(self) -> None:
        """埋め込みインデックスをロードまたは構築"""
        cache_file = self.cache_dir / "manual_index.json"
        embeddings_file = self.cache_dir / "manual_embeddings.npy"
        
        if cache_file.exists() and embeddings_file.exists():
            print("[rag] Loading cached embeddings")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunks = data["chunks"]
            self.embeddings = np.load(embeddings_file)
            print(f"[rag] Loaded {len(self.chunks)} chunks from cache")
        else:
            print("[rag] Building new embeddings (this may take a few minutes)")
            self._build_index()
            
            # キャッシュに保存
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"chunks": self.chunks}, f, ensure_ascii=False, indent=2)
            np.save(embeddings_file, self.embeddings)
            print(f"[rag] Saved {len(self.chunks)} chunks to cache")
    
    def _build_index(self) -> None:
        """PDFからインデックスを構築"""
        # PDFからテキスト抽出
        text = self._extract_text_from_pdf()
        
        # チャンクに分割
        self.chunks = self._split_into_chunks(text)
        
        # 埋め込み生成
        print(f"[rag] Generating embeddings for {len(self.chunks)} chunks...")
        self.embeddings = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    
    def _extract_text_from_pdf(self) -> str:
        """PDFからテキストを抽出"""
        if not os.path.exists(self.pdf_path):
            print(f"[rag] Warning: PDF not found at {self.pdf_path}")
            return ""
        
        print(f"[rag] Extracting text from {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        text_parts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())
        
        doc.close()
        full_text = "\n".join(text_parts)
        print(f"[rag] Extracted {len(full_text)} characters from {len(doc)} pages")
        return full_text
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """テキストをチャンクに分割"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # 文の途中で切れないように調整
            if end < len(text):
                # 最後の句点や改行を探す
                last_period = chunk.rfind("。")
                last_newline = chunk.rfind("\n")
                split_point = max(last_period, last_newline)
                
                if split_point > self.chunk_size // 2:
                    chunk = chunk[:split_point + 1]
                    end = start + split_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if len(c) > 50]  # 短すぎるチャンクを除外
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        クエリに関連するマニュアルセクションを検索
        
        Args:
            query: 検索クエリ（例: "脚が動かない原因"）
            top_k: 返す結果の数
        
        Returns:
            [(chunk_text, similarity_score), ...] のリスト
        """
        if not self.enabled or self.embeddings is None:
            return []
        
        # クエリの埋め込み生成
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # コサイン類似度計算
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 上位k件を取得
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(similarities[idx])))
        
        return results
    
    def get_context_for_diagnosis(
        self,
        leg_id: str,
        symptoms: List[str],
        sensor_data: Dict[str, float],
    ) -> str:
        """
        診断に関連するマニュアルコンテキストを取得
        
        Args:
            leg_id: 脚ID
            symptoms: 症状のリスト（例: ["動かない", "埋まっている"]）
            sensor_data: センサーデータ
        
        Returns:
            マニュアルからの関連テキスト
        """
        if not self.enabled:
            return "（マニュアル情報なし）"
        
        # クエリを構築
        query_parts = [f"{leg_id}脚"]
        query_parts.extend(symptoms)
        
        if sensor_data.get("spot_can", 0) < 0.3:
            query_parts.append("自己診断 低スコア")
        if sensor_data.get("drone_can", 0) < 0.3:
            query_parts.append("外部観測 動作不良")
        
        query = " ".join(query_parts)
        
        # 検索
        results = self.search(query, top_k=2)
        
        if not results:
            return "（関連するマニュアル情報が見つかりませんでした）"
        
        # コンテキストを整形
        context_parts = ["【Spotマニュアルからの関連情報】\n"]
        for i, (chunk, score) in enumerate(results, 1):
            context_parts.append(f"\n関連度 {score:.2f}:")
            context_parts.append(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        return "\n".join(context_parts)


# シングルトンインスタンス（メモリ節約）
_manual_rag_instance: Optional[ManualRAG] = None


def get_manual_rag(
    pdf_path: str = "/home/kk21053/sotuken/Spot_IFU-v2.1.2-ja.pdf",
    cache_dir: str = "data/manual_embeddings",
) -> Optional[ManualRAG]:
    """ManualRAGのシングルトンインスタンスを取得"""
    global _manual_rag_instance
    
    if _manual_rag_instance is None:
        try:
            _manual_rag_instance = ManualRAG(pdf_path=pdf_path, cache_dir=cache_dir)
        except Exception as e:
            print(f"[rag] Failed to initialize ManualRAG: {e}")
            return None
    
    return _manual_rag_instance
