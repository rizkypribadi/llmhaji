"""
Hallucination Mitigation Layer untuk Virtual Muthawwif Cerdas
==============================================================

Modul ini mengimplementasikan layer keamanan pasca-generasi untuk mencegah halusinasi
dan memastikan jawaban LLM benar-benar didukung oleh konteks yang diretrieve.

Komponen Verifikasi:
1. Entailment Check: Menggunakan NLI model untuk verifikasi logical entailment
2. Citation Verification: Memeriksa keberadaan dalil/kutipan dalam konteks
3. Semantic Overlap: Menghitung kemiripan semantik antara answer dan context
4. Keyword Validation: Memastikan istilah fikih penting ada di konteks

Referensi Penelitian:
[14], [34] - Hallucination Mitigation Layer sebagai fokus keselamatan (safety focus)
"""

import re
from typing import Dict, List, Optional, Tuple
from sentence_transformers import CrossEncoder, SentenceTransformer, util
import numpy as np


class HallucinationChecker:
    """
    Lapisan verifikasi silang untuk mendeteksi dan mencegah halusinasi LLM.
    
    Implementasi sesuai arsitektur penelitian yang mensyaratkan:
    - Verifikasi silang terhadap teks generatif
    - Kepatuhan penuh pada kebenaran yurisprudensi
    - Valid verified berdasarkan dalil-dalil hukum Islam otoritatif
    """
    
    # Threshold konfigurasi
    ENTAILMENT_THRESHOLD = 0.65      # Skor NLI minimum untuk entailment
    OVERLAP_THRESHOLD = 0.40          # Skor semantic overlap minimum
    CITATION_PATTERNS = [
        r'(QS\.|Quran|Al-Qur\'an|Surah)\s+[\w\s:-]+',  # Referensi Al-Quran
        r'(HR\.|Hadits|Hadis|Riwayat)\s+[\w\s:-]+',    # Referensi Hadits
        r'(Imam|Ulama|Syekh)\s+[\w\s-]+',              # Referensi ulama
        r'(rukun|wajib|sunnah|makruh|haram)',          # Istilah fikih
    ]
    
    # Keyword fikih haji/umrah yang harus validated
    FIQH_KEYWORDS = {
        'ihram', 'thawaf', 'sa\'i', 'sai', 'wukuf', 'arafah', 'mabit', 'mina', 
        'muzdalifah', 'jumrah', 'tahallul', 'dam', 'fidyah', 'miqat', 'rukun',
        'wajib', 'sunnah', 'dalil', 'hadits', 'quran', 'rasulullah', 'nabi'
    }
    
    def __init__(self, 
                 nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Inisialisasi Hallucination Checker dengan model verifikasi.
        
        Args:
            nli_model_name: Model untuk Natural Language Inference (entailment check)
            embedding_model_name: Model untuk semantic similarity calculation
        """
        print("[*] Inisialisasi Hallucination Mitigation Layer...")
        
        try:
            # NLI model untuk entailment verification
            print(f"|-- Loading NLI Model: {nli_model_name}")
            self.nli_model = CrossEncoder(nli_model_name)
            
            # Embedding model untuk semantic overlap
            print(f"|-- Loading Embedding Model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            
            print("[✔] Hallucination Checker siap!")
            
        except Exception as e:
            print(f"[!] Error inisialisasi Hallucination Checker: {e}")
            raise
    
    def verify_answer(self, 
                     answer: str, 
                     context: str,
                     query: Optional[str] = None,
                     sources: Optional[List[str]] = None) -> Dict:
        """
        Verifikasi komprehensif apakah jawaban LLM didukung oleh konteks.
        
        Args:
            answer: Jawaban yang di-generate oleh LLM
            context: Konteks dari dokumen yang diretrieve
            query: (Optional) Pertanyaan asli user
            sources: (Optional) Daftar nama file sumber
        
        Returns:
            Dict dengan struktur:
            {
                "is_valid": bool,           # Apakah jawaban lolos verifikasi
                "confidence": float,        # Confidence score (0-1)
                "entailment_score": float,  # Skor NLI entailment
                "overlap_score": float,     # Skor semantic overlap
                "has_citations": bool,      # Apakah ada dalil/kutipan
                "has_fiqh_terms": bool,     # Apakah mengandung istilah fikih
                "validation_details": str,  # Penjelasan hasil verifikasi
                "mitigation_applied": bool  # Apakah perlu mitigasi
            }
        """
        print("[*] Memulai verifikasi hallucination...")
        
        # Normalisasi input
        answer = answer.strip()
        context = context.strip()
        
        # 1. Entailment Check (apakah context mendukung answer)
        entailment_score = self._check_entailment(context, answer)
        print(f"|-- Entailment Score: {entailment_score:.4f}")
        
        # 2. Citation Verification (apakah ada dalil/referensi)
        has_citations, citation_details = self._verify_citations(answer, context)
        print(f"|-- Has Citations: {has_citations} ({citation_details})")
        
        # 3. Semantic Overlap (kemiripan semantik)
        overlap_score = self._calculate_semantic_overlap(answer, context)
        print(f"|-- Semantic Overlap: {overlap_score:.4f}")
        
        # 4. Fiqh Keyword Validation
        has_fiqh_terms, fiqh_details = self._validate_fiqh_keywords(answer, context)
        print(f"|-- Fiqh Terms Validated: {has_fiqh_terms} ({fiqh_details})")
        
        # 5. Special case handling
        is_greeting_or_fallback = self._is_safe_fallback(answer)
        
        # Decision Logic
        is_valid, validation_details = self._make_decision(
            entailment_score=entailment_score,
            overlap_score=overlap_score,
            has_citations=has_citations,
            has_fiqh_terms=has_fiqh_terms,
            is_greeting_or_fallback=is_greeting_or_fallback
        )
        
        # Hitung confidence score
        confidence = self._calculate_confidence(
            entailment_score, overlap_score, has_citations, has_fiqh_terms
        )
        
        result = {
            "is_valid": is_valid,
            "confidence": confidence,
            "entailment_score": entailment_score,
            "overlap_score": overlap_score,
            "has_citations": has_citations,
            "has_fiqh_terms": has_fiqh_terms,
            "validation_details": validation_details,
            "mitigation_applied": not is_valid,
            "citation_details": citation_details,
            "fiqh_details": fiqh_details
        }
        
        print(f"[*] Verifikasi selesai: {'✅ VALID' if is_valid else '❌ HALLUCINATION DETECTED'}")
        return result
    
    def _check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Gunakan NLI model untuk cek apakah premise (context) mendukung hypothesis (answer).
        
        Returns:
            Entailment score (0-1), semakin tinggi semakin baik
        """
        try:
            # NLI model output 3 class probabilities setelah softmax
            # DeBERTa-v3 NLI menggunakan label mapping:
            # Index 0: contradiction
            # Index 1: entailment  ← yang kita butuhkan
            # Index 2: neutral
            scores = self.nli_model.predict([[premise, hypothesis]], apply_softmax=True)
            
            if isinstance(scores, np.ndarray) and len(scores.shape) > 1:
                # Ambil skor entailment (index 1 untuk DeBERTa NLI)
                entailment_score = float(scores[0][1])
            else:
                # Fallback jika format berbeda
                entailment_score = float(scores[0]) if hasattr(scores[0], '__float__') else 0.5
            
            return entailment_score
            
        except Exception as e:
            print(f"[!] Error dalam entailment check: {e}")
            return 0.5  # Neutral score jika error
    
    def _verify_citations(self, answer: str, context: str) -> Tuple[bool, str]:
        """
        Verifikasi apakah answer mengandung dalil/referensi yang ada di context.
        
        Returns:
            (has_valid_citations, details_string)
        """
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        found_patterns = []
        
        for pattern in self.CITATION_PATTERNS:
            matches_in_answer = re.findall(pattern, answer, re.IGNORECASE)
            if matches_in_answer:
                # Cek apakah dalil ini juga ada di context
                for match in matches_in_answer:
                    if match.lower() in context_lower:
                        found_patterns.append(match)
        
        # Cek juga keyword umum seperti "maaf", "tidak menemukan" = fallback answer
        is_fallback = any(keyword in answer_lower for keyword in [
            'maaf', 'tidak menemukan', 'belum menemukan', 'tidak ada', 'konsultasikan'
        ])
        
        if is_fallback:
            return True, "Fallback answer (safe)"
        
        if found_patterns:
            return True, f"Found {len(found_patterns)} citations: {', '.join(found_patterns[:3])}"
        
        # Jika tidak ada dalil eksplisit tapi jawaban pendek/umum, anggap aman
        if len(answer.split()) < 20:
            return True, "Short answer (low risk)"
        
        return False, "No citations found in context"
    
    def _calculate_semantic_overlap(self, answer: str, context: str) -> float:
        """
        Hitung semantic similarity antara answer dan context menggunakan cosine similarity.
        
        Returns:
            Overlap score (0-1)
        """
        try:
            # Encode kedua teks
            answer_embedding = self.embedding_model.encode(answer, convert_to_tensor=True)
            context_embedding = self.embedding_model.encode(context, convert_to_tensor=True)
            
            # Hitung cosine similarity
            similarity = util.cos_sim(answer_embedding, context_embedding)
            
            return float(similarity[0][0])
            
        except Exception as e:
            print(f"[!] Error dalam semantic overlap calculation: {e}")
            return 0.5
    
    def _validate_fiqh_keywords(self, answer: str, context: str) -> Tuple[bool, str]:
        """
        Validasi apakah istilah fikih dalam answer juga ada di context.
        
        Returns:
            (is_valid, details_string)
        """
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        # Ekstrak keyword fikih dari answer
        fiqh_in_answer = [kw for kw in self.FIQH_KEYWORDS if kw in answer_lower]
        
        if not fiqh_in_answer:
            # Tidak ada istilah fikih, bisa jadi greeting/fallback
            return True, "No specific fiqh terms (general answer)"
        
        # Cek apakah keyword tersebut juga ada di context
        validated_keywords = [kw for kw in fiqh_in_answer if kw in context_lower]
        
        validation_rate = len(validated_keywords) / len(fiqh_in_answer) if fiqh_in_answer else 1.0
        
        if validation_rate >= 0.7:  # 70% keyword harus ada di context
            return True, f"{len(validated_keywords)}/{len(fiqh_in_answer)} keywords validated"
        else:
            return False, f"Only {len(validated_keywords)}/{len(fiqh_in_answer)} keywords found in context"
    
    def _is_safe_fallback(self, answer: str) -> bool:
        """
        Deteksi apakah jawaban adalah safe fallback message (misal: "Maaf, saya tidak tahu").
        """
        answer_lower = answer.lower()
        
        safe_patterns = [
            'maaf', 'mohon maaf', 'tidak menemukan', 'belum menemukan',
            'tidak ada informasi', 'konsultasikan', 'silakan tanya',
            'assalamu', 'wa\'alaikum', 'halo', 'hai', 'terima kasih'
        ]
        
        return any(pattern in answer_lower for pattern in safe_patterns)
    
    def _make_decision(self, 
                      entailment_score: float,
                      overlap_score: float,
                      has_citations: bool,
                      has_fiqh_terms: bool,
                      is_greeting_or_fallback: bool) -> Tuple[bool, str]:
        """
        Logika keputusan apakah jawaban valid atau hallucination.
        
        Returns:
            (is_valid, validation_details)
        """
        # Safe fallback selalu valid
        if is_greeting_or_fallback:
            return True, "Safe fallback/greeting message"
        
        # Kriteria ketat untuk jawaban substantif
        if entailment_score >= self.ENTAILMENT_THRESHOLD:
            if overlap_score >= self.OVERLAP_THRESHOLD or has_citations:
                return True, f"Strong entailment ({entailment_score:.2f}) + good overlap/citations"
            elif has_fiqh_terms:
                return True, f"Good entailment ({entailment_score:.2f}) + validated fiqh terms"
            else:
                return False, f"Entailment OK but low overlap ({overlap_score:.2f}) and no citations"
        
        # Moderate entailment dengan dukungan kuat
        elif entailment_score >= 0.50:
            if has_citations and overlap_score >= self.OVERLAP_THRESHOLD:
                return True, f"Moderate entailment with strong support"
            else:
                return False, f"Moderate entailment ({entailment_score:.2f}) lacks strong support"
        
        # Low entailment = hallucination risk
        else:
            return False, f"Low entailment score ({entailment_score:.2f}) - hallucination risk"
    
    def _calculate_confidence(self,
                            entailment_score: float,
                            overlap_score: float,
                            has_citations: bool,
                            has_fiqh_terms: bool) -> float:
        """
        Hitung confidence score gabungan (0-1).
        """
        # Weighted combination
        confidence = (
            entailment_score * 0.50 +      # 50% bobot entailment
            overlap_score * 0.30 +         # 30% bobot semantic overlap
            (0.10 if has_citations else 0) +  # 10% bonus citations
            (0.10 if has_fiqh_terms else 0)   # 10% bonus fiqh validation
        )
        
        return min(1.0, max(0.0, confidence))
    
    def get_safe_fallback_message(self) -> str:
        """
        Kembalikan pesan fallback yang aman jika deteksi hallucination.
        """
        return (
            "Maaf 🙏, saya belum menemukan informasi yang cukup akurat untuk menjawab "
            "pertanyaan tersebut dalam basis pengetahuan saya. Untuk kepastian ibadah, "
            "silakan konsultasikan dengan ustadz atau pembimbing haji Anda. "
            "Jazakallahu khairan atas pengertiannya."
        )


# Singleton instance untuk efisiensi (lazy loading)
_hallucination_checker_instance: Optional[HallucinationChecker] = None


def get_hallucination_checker(
    nli_model: str = "cross-encoder/nli-deberta-v3-base",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
) -> HallucinationChecker:
    """
    Factory function untuk mendapatkan singleton instance HallucinationChecker.
    """
    global _hallucination_checker_instance
    
    if _hallucination_checker_instance is None:
        _hallucination_checker_instance = HallucinationChecker(
            nli_model_name=nli_model,
            embedding_model_name=embedding_model
        )
    
    return _hallucination_checker_instance
