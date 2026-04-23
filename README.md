# Gugak-VocalSet

**A High-quality Korean Traditional Singing Dataset with Sigimsae Annotations**

> This repository accompanies our ISMIR 2026 submission.
> **Anonymous release for double-blind review.** Author names, affiliations, and any identifying metadata have been removed from the code, documentation, and dataset packaging.

---

## Overview

Gugak-VocalSet is the first publicly available audio dataset for Korean traditional singing, with dense annotations of *sigimsae* — the continuous pitch-level vocal expression that constitutes the melodic identity of Korean traditional vocal music.

- **24.5 hours** of unaccompanied solo vocal recordings
- **300 pieces / 305 tracks** of representative repertoire curated by the National Gugak Center
- **20 professional singers**, recorded in a controlled studio environment with fixed key and tempo
- **~69,000 audio-grounded sigimsae annotations** following a unified **17-type ontology** organized by functional position (onset / mid-note / offset / transition / accent / vibrato)
- Rich metadata: genre hierarchy, lyrics–audio alignment, Western key, jangdan, tempo, bilingual captions, expert mood & timbre tags

---

## Sigimsae Ontology

The 17 sigimsae types are organized by **functional position within the melodic line**:

| Group | Types (Korean label → English) |
|---|---|
| **Onset** *(앞시김새)* | 밀어내기 → onset asc. slide · 꺾어내기 → onset desc. flick · 감아내기(위) → onset upper turn · 감아내기(아래) → onset lower turn |
| **Vibrato** *(요성)* | 보통 요성 → standard vibrato · 격한 요성 → wide vibrato · 아래로 향하는 요성 → lower-centered vibrato · 흘러내리는 요성 → desc.-tail vibrato |
| **Mid-note** *(중간 시김새)* | 굴려내기(위) → mid-note upper roll · 굴려내기(아래) → mid-note lower roll |
| **Offset** *(뒷시김새)* | 추성 *(chuseong)* → offset asc. slide · 퇴성 *(toeseong)* → offset desc. slide · 끝굴려내기(위) → offset roll-back · 끝굴려내기(아래) → offset lower roll-away · 끊어내기 → offset sharp cut |
| **Accent** *(강조 시김새)* | 전성 *(jeonseong)* → upward-accented vibrato |
| **Transition** *(경과 시김새)* | 채올리기 → transition upward snap |

See the paper for notation symbols, contour illustrations, and examples.

---

## Dataset Files

Each track has a stable ID of the form

```
KC_TM_{JC|MF}_{GJ|PR|MY|PS}_P{######}
```

| Segment | Meaning |
|---|---|
| `KC_TM` | fixed dataset prefix |
| `JC` / `MF` | top-level genre — **정악 (Jeongak)** / **민속악 (Minsogak)** |
| `GJ` / `PR` / `MY` / `PS` | mid-level genre — **궁중음악 (GJ)** / **풍류음악 (PR)** / **민요 (MY)** / **판소리 (PS)** |
| `P######` | 6-digit zero-padded track index |

For each track we release (i) one row in a track-level metadata CSV and (ii) one JSON file holding the time-aligned annotations.

---

## Track-level Metadata (`metadata.csv`)

Each row describes one track; the track ID column keys into the annotation JSON filename.

| Column | Type | Description |
|---|---|---|
| `정제데이터 파일명` *(track ID)* | string | Stable track identifier, e.g. `KC_TM_JC_PR_S000001`. Matches the annotation JSON filename (with `S` → `P`). |
| `분류_업데이트` *(category)* | categorical | Fine-grained style category: 정가, 궁중악, 풍류악, 판소리, 민요. |
| `장르` *(top-level genre)* | categorical | **정악** or **민속악**. |
| `하위장르` *(mid-level genre)* | categorical | 궁중음악 · 풍류음악 · 판소리 · 민요. |
| `세부장르(optional)` *(sub-genre)* | categorical | Genre-specific sub-label (23 values). For pansori: the narrative (e.g. 심청가, 춘향가). For folk songs: representative regional origin (남도, 경기, 서도, 병창). |
| `곡명` *(title)* | string | Piece title. |
| `가창자` *(singer)* | string | Performer name (20 singers). |
| `가창자 성별` *(gender)* | categorical | **남창** / **여창**. |
| `최종 서양조성` *(Western key)* | string | Western-music key tag (e.g. `Ab / Fm`), assigned by Western-music-trained annotators for general-purpose use. |
| `장단` *(jangdan)* | string | Rhythmic cycle (17 distinct values across the corpus). |
| `최종 박자` *(time signature)* | string | e.g. `48/8`, `12/8`. |
| `최종 템포` *(tempo)* | float | BPM. |
| `moods_all` | string | Expert mood tags with vote counts, comma-separated (80-tag vocabulary). Example: `기품 있는(2), 잔잔한(2), ...`. |
| `timbres_all` | string | Expert timbre tags with vote counts, comma-separated (38-tag vocabulary). |
| `캡션(한국어)` *(caption, KR)* | string | Korean prose caption describing the piece. Duplicated in the JSON as `caption_ko`. |
| `캡션(영어)` *(caption, EN)* | string | English prose caption. Duplicated in the JSON as `caption_en`. |
| `가사` *(lyrics, full text)* | string | Full transcribed lyrics of the piece, line-separated. |
| `JSON_가사_모음` *(lyrics segments)* | string | Lyrics split into phrase-level segments (`/`-separated). Aligned 1-to-1 with `lyrics_regions` in the JSON. |
| `가사세그먼트` *(n-segments)* | int | Number of phrase segments. |
| `오디오_길이(초)` *(duration)* | float | Audio duration in seconds. Mean 286.7 s, median 260.1 s. |

> **Not released.** Internal fields (`Old ID`, `예상금액`, `길이수정됨`, `배정_라벨러`, `라벨러이름`) are stripped before distribution.

---

## Per-track Annotation JSON

Each track has a JSON file named after its track ID, e.g. `KC_TM_JC_GJ_P000074.json`, containing the time-aligned lyrics and sigimsae annotations plus bilingual captions:

```jsonc
{
  "info": {
    "identifier": "KC_TM_JC_GJ_P000074",   // matches the filename (sans .json)
    "source_id":  "KC_TM_JC_GJ_M000074"    // original studio master ID (same index, M prefix)
  },
  "annotation": {
    "lyrics_regions": [
      {
        "lyrics_id":  "L001",              // L###, unique within the track
        "start_sec":  1.693031553127439,   // phrase start (seconds)
        "end_sec":   15.494926962443781,   // phrase end   (seconds)
        "lyrics":    "황의상제 구민지막"   // text as actually sung
      }
      // ... 7–20 s phrases, delimited by jangdan structure
    ],
    "sigimsage_regions": [
      {
        "sigimsage_id":    "S001",              // S###, unique within the track
        "start_sec":       3.6434831084847805,
        "end_sec":         3.8613000334485466,
        "sigimsage_types": ["추성"]             // one or more labels (see ontology above)
      }
      // ... typically 150–400 events per track
    ],
    "caption_ko": "귀인은 종묘제례악 중 하나로, …",
    "caption_en": "Gwi-in is one of the movements in Jerye Ritual Music, …"
  }
}
```

### Field reference

**`info`**

| Key | Type | Description |
|---|---|---|
| `identifier` | string | Track ID; matches the filename. |
| `source_id`  | string | Master recording ID; identical numeric part with an `M` prefix. |

**`annotation.lyrics_regions[]`** — phrase-level lyrics–audio alignment (7–20 s per region, delimited by jangdan). The regions are in 1-to-1 correspondence with the `JSON_가사_모음` column in `metadata.csv`.

| Key | Type | Description |
|---|---|---|
| `lyrics_id` | string | Per-track unique, zero-padded (`L001`, `L002`, …). |
| `start_sec` / `end_sec` | float | Region bounds in seconds from the start of the audio. |
| `lyrics` | string | Transcribed text exactly as sung by the performer. |

**`annotation.sigimsage_regions[]`** — event-level sigimsae annotations.
*(The key is spelled `sigimsage_*` in the JSON; the label vocabulary and paper text use the romanization **sigimsae**.)*

| Key | Type | Description |
|---|---|---|
| `sigimsage_id` | string | Per-track unique, zero-padded (`S001`, `S002`, …). |
| `start_sec` / `end_sec` | float | Event bounds in seconds (mean duration ≈ 0.47 s). |
| `sigimsage_types` | list[string] | One or more Korean labels drawn from the 17-type ontology above. ~4.9 % of regions carry **two or more labels** (multi-label). |

**`annotation.caption_ko` / `caption_en`** — free-text bilingual captions, mirroring the corresponding columns in `metadata.csv`.

### Label statistics

Across 305 tracks / 69,127 regions:

| Label | Count | Label | Count |
|---|---:|---|---:|
| 꺾어내기        | 15,713 | 끊어내기              | 1,743 |
| 보통 요성       | 11,158 | 아래로 향하는 요성    | 1,398 |
| 굴려내기(위)    |  9,749 | 흘러내리는 요성        | 1,110 |
| 추성             |  6,391 | 끝굴려내기(아래)       | 1,075 |
| 격한 요성        |  5,524 | 끝굴려내기(위)         |   440 |
| 감아내기(위)     |  4,496 | 감아내기(아래)         |   431 |
| 밀어내기         |  3,642 |                        |       |
| 굴려내기(아래)   |  2,606 |                        |       |
| 채올리기         |  2,605 |                        |       |
| 퇴성             |  2,602 |                        |       |
| 전성             |  1,825 |                        |       |

Multi-label regions: **3,364 / 69,127 (4.9 %)**.

---

## Dataset Release

This repository serves as the anonymous companion site during review. The audio and annotation files will be released upon publication.

---

## License

- **Code:** released under the MIT License (see `LICENSE`).
- **Dataset:** distributed under CC BY-NC-SA 4.0. Recordings were made under agreements with the performers and the National Gugak Center.

---

## Citation

Citation information will be provided upon publication. During the review period, please refer to this repository only as the anonymous companion to the ISMIR 2026 submission titled *"Gugak-VocalSet: A High-quality Korean Traditional Singing Dataset with Sigimsae Annotations."*

---

## Anonymity Notice for Reviewers

Please do not attempt to de-anonymize this repository. We have scrubbed author names, affiliations, acknowledgements, and identifying commit metadata. Any remaining identifying artifacts are unintentional — we would be grateful if you could notify the program chairs so they can be removed.
