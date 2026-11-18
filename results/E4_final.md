# E4 — verification layer pada run final

Jawaban dinilai: 129 (jawaban kosong dikeluarkan: 4); max_length NLI = 512

**Flag rate:** 65/129 = 50.4%

Pasangan NLI yang terpotong (> 512 token): 129/129; median token premis 1437, hipotesis 88

## Acuan: RAGAS faithfulness < 1 (n = 129; positif = 20, negatif = 109)

| | flagged | lolos |
|---|---|---|
| positif (halusinasi) | TP=8 | FN=12 |
| negatif | FP=57 | TN=52 |

- Recall = 8/20 = 0.400
- FPR = 57/109 = 0.523
- Precision = 8/65 = 0.123

## Kontribusi pemeriksaan

- Lolos via aturan safe-fallback/greeting (substring): 7
- NLI ≥ 0,65: 57; 0,50–0,65: 4; < 0,50: 68
- Overlap ≥ 0,40: 124 (median 0.768)
- Citation check lolos: 102
- Keyword check lolos: 129

Alasan keputusan (validation_details, dikelompokkan):

- Low entailment score: 63
- Strong entailment: 55
- Safe fallback/greeting message: 7
- Moderate entailment with strong support: 2
- Moderate entailment: 2

## Threshold sweep τ_NLI (acuan RAGAS faithfulness < 1)

| τ_NLI | flag rate | recall | FPR |
|---|---|---|---|
| 0.30 | 0.380 | 0.350 | 0.385 |
| 0.35 | 0.426 | 0.350 | 0.440 |
| 0.40 | 0.465 | 0.350 | 0.486 |
| 0.45 | 0.473 | 0.350 | 0.495 |
| 0.50 | 0.488 | 0.350 | 0.514 |
| 0.55 | 0.504 | 0.400 | 0.523 |
| 0.60 | 0.504 | 0.400 | 0.523 |
| 0.65 | 0.504 | 0.400 | 0.523 |
| 0.70 | 0.512 | 0.450 | 0.523 |
| 0.75 | 0.519 | 0.500 | 0.523 |
| 0.80 | 0.519 | 0.500 | 0.523 |
| 0.85 | 0.519 | 0.500 | 0.523 |
| 0.90 | 0.527 | 0.500 | 0.532 |

Catatan: ambang moderat 0,50 di _make_decision tertanam di kode (bukan atribut), sehingga untuk τ < 0,50 cabang 'moderat' tidak pernah aktif.