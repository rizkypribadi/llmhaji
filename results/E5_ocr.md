# E5 — CER/WER OCR

Sumber GT: 05 Manasik Haji dan Umrah - Terjemah Manhajus Salikin.pdf (lapisan teks digital). N = 25 halaman, seed 42.
Halaman: [6, 7, 8, 10, 11, 17, 18, 19, 22, 33, 34, 37, 38, 40, 41, 43, 44, 49, 53, 58, 62, 63, 69, 74, 76]

| Setelan | Cakupan | CER mean | CER (bobot char) | CER median | WER mean | WER median | karakter kontrol RTL (total) |
|---|---|---|---|---|---|---|---|
| ind@300 | latin_raw | 4.5893 | 0.4165 | 0.1104 | 3.2984 | 0.1781 | 0 |
| ind@300 | latin_clean | 0.8154 | 0.1268 | 0.1197 | 0.5947 | 0.2143 | 0 |
| ind@300 | penuh_raw | 0.3566 | 0.3553 | 0.1522 | 0.4893 | 0.3714 | 0 |
| ind+ara@300 | latin_raw | 0.2159 | 0.0488 | 0.0561 | 0.3247 | 0.1250 | 194 |
| ind+ara@300 | latin_clean | 0.2471 | 0.0899 | 0.1300 | 0.3640 | 0.2083 | 194 |
| ind+ara@300 | penuh_raw | 0.1442 | 0.1456 | 0.0920 | 0.2667 | 0.2800 | 194 |
| ind@200 | latin_raw | 4.7514 | 0.4096 | 0.0611 | 3.3650 | 0.1237 | 0 |
| ind@200 | latin_clean | 0.7575 | 0.1103 | 0.1031 | 0.5435 | 0.1803 | 0 |
| ind@200 | penuh_raw | 0.3578 | 0.3563 | 0.1508 | 0.4885 | 0.3714 | 0 |
| ind@150 | latin_raw | 4.6166 | 0.4057 | 0.0713 | 3.3465 | 0.1111 | 0 |
| ind@150 | latin_clean | 0.6064 | 0.1119 | 0.1053 | 0.5275 | 0.1806 | 0 |
| ind@150 | penuh_raw | 0.3620 | 0.3607 | 0.1607 | 0.4955 | 0.3714 | 0 |

Catatan: halaman dirender dari PDF digital (bersih, tanpa derau pindai), sehingga angka ini adalah batas bawah (optimistis) dibanding pindaian nyata seperti Kemenhaj.