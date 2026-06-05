# Analisi del workspace — `latent_space_compression_research`

> Resoconto generato il **2026-06-04** da un'analisi statica (sola lettura) del repository.
> Nessun file del progetto è stato modificato: questo documento è l'unico artefatto creato.
> Branch: `main` · file tracciati da git: **80**.

---

## 1. In una frase

Studio di fattibilità che **quantizza il canale latente inter-agente** del sistema multi-agente
[RecursiveMAS](https://github.com/RecursiveMAS/RecursiveMAS) (configurazione *Sequential-Light*) usando il
**nucleo MSE-ottimale di [TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) — rotazione di Haar
*data-oblivious* + quantizzatore scalare Lloyd-Max — e misura l'effetto sull'accuratezza in `math500`.

Non è "TurboQuant migliora RecursiveMAS": è la domanda **"quanto è comprimibile quello specifico canale di
comunicazione, e con quali conseguenze a valle?"**

---

## 2. Il risultato scientifico principale

**Variant B comprime il canale latente inter-agente 4×–16× senza variazione di accuratezza misurabile sotto
decoding *sampled*.**

| Bits/coord | Compressione | Accuratezza n=250 | Δ vs baseline | z (2-prop) | p |
|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 1× | 75.2% | — | — | — |
| 8 | 4× | 78.4% | +3.2 pp | 0.83 | > 0.4 ✓ |
| 4 | 8× | 76.8% | +1.6 pp | 0.41 | > 0.5 ✓ |
| 2 | 16× | 75.2% | 0.0 pp | 0.00 | identico ✓ |

A 2-bit l'accuratezza è **bit-for-bit identica** al baseline (188/250 in entrambi i casi). Misurato su
math500, n=250, seed=42, Kaggle T4 con `--dtype float32`, decoding *sampled*.

### La sfumatura onesta (il cuore intellettuale del progetto)

Il repository è notevolmente **rigoroso e auto-critico**. Il titolo "no measurable change" è esplicitamente
condizionato:

- Sotto decoding **greedy**, un test di equivalenza appaiato più severo (**TOST ±2pp**, n=250) è
  **INCONCLUSIVO** (Δ = −2.0 pp, non significativo).
- La perturbazione a 4-bit **cambia il testo del ragionamento generato in ~88% dei problemi**, pur
  **preservando la risposta finale**.
- Da qui la tesi più originale e la domanda-guida della ricerca:
  **"answer-preserving ≠ trajectory-preserving"** — perché la risposta sopravvive mentre la traiettoria
  cambia, e *quando questo si rompe* (ipotesi: si rompe dove la traiettoria *è* l'output, es. generazione di
  codice / tool calls).

"No detected change" è correttamente trattato come **non-prova di equivalenza**, non come "lossless".

---

## 3. Struttura del repository

```
.
├── README.md                  ← overview pubblica (TL;DR + riproduzione)
├── AGENTS.md                  ← istruzioni per agenti AI (invarianti, layout, hazard)
├── ROADMAP.md                 ← lavoro futuro (8-cell matrix, QJL, ecc.)  [untracked]
├── REPRODUCIBILITY.md         ← riproduzione end-to-end (verify → analyze → rerun)
├── LICENSE                    ← All Rights Reserved (source-visible, NON open-source)
├── requirements.txt           ← env locale (numpy, scipy, matplotlib, torch 2.12, pytest)
├── src/                       ← libreria (quantizzatori + patch + metriche)  ~1.0k LOC
├── tests/                     ← 7 file, 99 funzioni test, 1452 LOC
├── experiments/               ← 5 cartelle attive, runner cloud  ~3.6k LOC Python
├── docs/                      ← TUTTA la documentazione (reports, design, operations, figures)
├── writeup/                   ← paper LaTeX (main.tex 467 righe + main.pdf + bib)
├── bin/                       ← wrapper CLI (kaggle, push_*.sh, verify_artifacts.py)
└── external/RecursiveMAS/     ← clone upstream (gitignored, 4.8 MB)
```

Dimensioni: `external` 4.4M · `docs` 612K · `experiments` 688K · `writeup` 364K · `tests` 192K · `src` 132K.

---

## 4. Codice sorgente (`src/`) — analisi modulo per modulo

Codice **pulito, ben documentato, type-hinted**, PyTorch puro senza assunzioni hardware (gira su CPU per i
test). Ogni funzione pubblica ha docstring.

### 4.1 Quantizzatori (`src/quantizers/`)

| Modulo | Ruolo | Algoritmo |
|---|---|---|
| `hadamard_uniform.py` | **Variant A** (screening) | RHT — `diag(±1)·FWHT` O(d log d) + quantizzatore uniforme simmetrico. Padding a potenza di 2. NON è TurboQuant fedele: è uno screen economico. |
| `turboquant_honest.py` | **Variant B** (il test scientifico) | Rotazione di Haar (QR di gaussiana, sign-corrected, O(d²)) + codebook **Lloyd-Max per N(0,1/d)** + normalizzazione L2 sulla sfera. È il nucleo MSE-ottimale di TurboQuant **senza** il residuo QJL. |

Dettagli notevoli di Variant B:
- Nearest-neighbor vettorizzato via `torch.bucketize` sui midpoint del codebook → O(N·d·log K) tempo,
  O(N·d) memoria (evita il tensore distanze `[N,d,K]`).
- La norma è preservata *esattamente* fuori dalla pipe quantizzata (la garanzia Beta vale solo sulla sfera).
- Nessuna modalità per-channel: il senso della rotazione di Haar è rendere le coordinate i.i.d. così che un
  unico codebook sia ottimale.

### 4.2 Adapter / patching (`src/adapters/patch.py`)

Monkey-patch **reversibile** di `CrossModelAdapter.forward` / `Adapter.forward` senza forkare l'upstream.
Proprietà progettate con cura:
- **Reversibile** (`patch_adapter` ritorna una `unpatch()` idempotente), tracciamento dei patch attivi via
  `WeakValueDictionary` + lock, rilevamento del doppio-patch.
- **dtype-preserving**: quantizza in fp32 per sicurezza numerica ma restituisce il dtype originale (cruciale —
  vedi §6 hardware).
- `QuantStats` accumula statistiche per-call (rMSE, cosine, norm_ratio) con memoria limitata, più un registry
  globale per la capture-mode nei kernel subprocessati.

> Nota di code-review: `unpatch_all()` è documentato come **non in grado di ripristinare davvero** senza le
> closure — è un semplice contatore/escape-hatch. È dichiarato esplicitamente nel codice, quindi è una scelta
> consapevole, ma è un'API leggermente trappola.

### 4.3 Metriche (`src/metrics/`)

| Modulo | Contenuto |
|---|---|
| `distortion.py` | `relative_mse`, `cosine`, `norm_ratio`, `inner_product_error` (quest'ultima predice la necessità di QJL). |
| `channel_fidelity.py` | `relative_l2`, `effective_rank` (participation ratio dello spettro singolare), `codebook_extreme_rate` (analogo Variant-B del clipping-rate), aggregatore `FidelityRun`. |
| `logit_metrics.py` | `per_position_mse/kl/js` all'egress (log-softmax-stabile, T=1.0), `summarize_pair` con gestione del troncamento quando REF e INT4 hanno lunghezze diverse. |
| `bootstrap.py` | **Bootstrap appaiato** + **TOST** (two one-sided t-test) con verdetto EQUIVALENT / NOT_EQUIVALENT / INCONCLUSIVE; gestione del caso degenere se=0; determinismo via `default_rng(seed)`. |

### 4.4 Utility (`src/utils/lloyd_max.py`)

Codebook Lloyd-Max per la marginale N(0,1/d), calcolato analiticamente con `scipy.integrate.quad` sul PDF
gaussiano, cache via `lru_cache(128)`. Rispecchia la reference `turboquant_ref` ed è verificato come oracolo
nei test.

---

## 5. Metodologia (notevolmente disciplinata)

Il design (`docs/RESEARCH.md`) è pre-registrato: **soglie e stage-gate fissati prima degli esperimenti**.

- **Due varianti in parallelo** per non confondere "scelta della rotazione" con "scelta del quantizzatore".
- **Gate sequenziali** (Gate 0 identity-sanity → 1A screen → 1B honest → 2 push-to-3-bit → 3 QAT → 4 QJL →
  5 systems) con una **matrice decisionale** esplicita (incluso il caso "Pass A / Fail B = bug, indaga").
- **Punto di inserzione giustificato**: il quantizzatore va *dopo* `ln_target`, perché quantizzare pre-LN
  sposterebbe la distribuzione di input di LN e confonderebbe il segnale.
- Ipotesi H0/H1/H2 dichiarate, con "un risultato negativo è comunque informativo".

---

## 6. Il filo conduttore hardware (finding indipendente di valore)

Una parte sostanziale del lavoro è la scoperta — root-caused con un esperimento a variabile singola — che
**RecursiveMAS Sequential-Light collassa silenziosamente a ~30% di accuratezza sulle GPU pre-Ampere** con il
default `--dtype auto`:

| GPU | sm | bf16 nativo | dtype | math500 |
|---|---|:---:|---|:---:|
| A100 | sm_80 | ✅ | auto (bf16) | **86%** ✓ |
| T4 | sm_75 | ❌ | auto → fallback | **30%** ❌ |
| T4 | sm_75 | ❌ | **float32 esplicito** | **84%** ✓ |
| P100 | sm_60 | ❌ | auto | **35%** ❌ |

Causa: il checkpoint è bf16; su hardware senza bf16 nativo PyTorch fa fallback e i ~80.000 matmul sequenziali
della ricorsione accumulano errori di range dinamico. **Forzare fp32 ripristina la correttezza.** Questo ha
invalidato 22 esperimenti P100 (ritrattati ma documentati) ed è stato "the project's biggest reproducibility
hazard". È anche la spiegazione del falso −19pp di Phase 0.F (artefatto di cast bf16↔fp32, non proprietà del
quantizzatore).

Questo è un *secondo* contributo metodologico autonomo, candidabile a una issue upstream.

---

## 7. Esperimenti e infrastruttura cloud (`experiments/`)

5 cartelle attive (~3.6k LOC Python di runner), tutto il clutter storico archiviato:

| Cartella | Scopo |
|---|---|
| `distortion_validation/` | rMSE per-link (sintetico + identity-check); §4.1 del paper. |
| `solver_diagnostic/` | sanity Solver-alone su math500 (83%). |
| `baseline_a100_modal/` | riproduzione baseline Modal A100. |
| `variant_b_ladder_t4_kaggle/` | **esperimento principale** — bit-rate ladder n=250 (Kaggle T4 fp32). |
| `fidelity_sweep/` | **Tier 2** — REF-vs-INT4 appaiato; due backend (`kernel_pkg`=Kaggle, `modal_pkg`=Modal A100) che condividono le *stesse* funzioni di patch testate. |

Pattern d'ingegneria solido: i runner cloud sono wrapper sottili che fanno **2 patch regex chirurgiche**
sull'upstream + iniezione runtime del quantizzatore; **la logica del modello e del quantizzatore non viene mai
modificata** (contratto "instrumentation only"). Verifica artefatti con manifest SHA256 (`bin/verify_artifacts.py`).

---

## 8. Test e qualità

- **99 funzioni `def test_`** su 7 file (1452 LOC). *(Nota: la documentazione cita "102 pass" in AGENTS.md e
  "109 pass" in README.md — la differenza col conteggio statico è plausibilmente dovuta a parametrizzazione,
  ma i due numeri tra loro non coincidono; vedi §11.)*
- Distribuzione: `test_fidelity_metrics` 25 · `test_fidelity_analyze` 20 · `test_hadamard_uniform` 14 ·
  `test_patch` 14 · `test_fidelity_kernel` 12 · `test_turboquant_honest` 12 ·
  `test_variant_b_ladder_analyze` 2.
- Convenzione dichiarata: ogni helper metrico/statistico ha almeno (a) test di determinismo, (b) edge-case,
  (c) sanity vs risposta nota. Variant B è verificato come **oracolo** contro `external/turboquant_ref` e
  contro i numeri pubblicati di TurboQuant Table 1 "fino alla terza cifra".
- I 9 patch regex sono testati con `compile()` + conteggio sostituzioni contro l'upstream reale → **validazione
  senza GPU** prima di spendere quota cloud. Approccio maturo.

---

## 9. Documentazione e write-up

Documentazione **eccezionalmente ricca** per un repo di ricerca:
- `docs/reports/` — 7 report numerati (01 sintetico → 06 headline → 07 fidelity), incluso un report
  esplicitamente **RETRACTED** con motivazione.
- `docs/design/` — architettura + design dell'indagine sul gap di 40pp.
- `docs/operations/` — log esperimenti + **audit di riproducibilità esterna**.
- `docs/figures/` — 4 PNG + script `_generate_figures.py` che li rigenera (regola dichiarata: niente ASCII
  bar chart nei report, sempre matplotlib→PNG).
- `writeup/main.tex` — paper completo (467 righe, PDF compilato): Introduction, Background, Method,
  Experiments (per-link distortion / sampled / greedy / depth-localization / reproducibility hazards),
  Discussion, Limitations, Conclusion.

---

## 10. Stato attuale e prossimi passi

**Chiuso/validato:** rMSE per-link allineato a TurboQuant; advisory hardware/dtype; ladder n=250 sampled.
**Attivo:** fidelity sweep Tier 2 su Modal A100 fp32 (localizzazione inner/outer **non risolta**; TOST greedy
inconcluso). **Aperto:** n=500 più potente; rinforzo del write-up.

`ROADMAP.md` (nuovo, ancora untracked) inquadra il futuro come una **matrice 8 celle** (2 stili × 4 benchmark,
1 fatta) con `mbppplus`/codice come test critico della domanda-guida, più: QJL residual, flip-churn analysis,
teacher-forced per-step KL, seed-robustness, e (Priority 1) generalità tra architetture.

---

## 11. Osservazioni critiche / incongruenze rilevate

Punti che un revisore — o tu, prima di pubblicare — potresti voler sistemare. Nessuno è bloccante; sono
questioni di **coerenza documentale**:

1. **Conteggio test divergente tra i documenti.** README.md dice "109 pass, 1 skip"; AGENTS.md dice "102 pass,
   1 skip"; il conteggio statico delle funzioni è 99. Conviene allineare a un'unica fonte di verità
   (idealmente l'output reale di `pytest --collect-only`).
2. **`docs/RESEARCH.md` è un palinsesto parzialmente stantio.** L'header dice ancora *"Phase 0.F in progress"*
   e la §13 riporta *"Does Variant B at 4-bit preserve accuracy? — answered NO, −19pp"*, mentre il finding
   corrente (README/AGENTS/§12.5.D) è che quel −19pp era un **artefatto dtype** e che a T4 fp32 è
   near-lossless. La spiegazione corretta *è presente* nel file, ma convive con la vecchia framing: un lettore
   nuovo può confondersi.
3. **Sezioni duplicate in `RESEARCH.md` §12.5.** Compaiono due intestazioni "D." (una "Phase 0.F result
   reframed", una "Phase 0.B RETRACTED") più una "E." quasi identica alla "D." successiva → evidente residuo
   di copia-incolla.
4. **`requirements.txt` cita `torch==2.12.0`.** Verificare che sia la versione realmente desiderata/esistente
   per la piattaforma di test (i runner cloud pinnano separatamente `torch==2.4.1+cu121`); il commento nel
   file invita comunque a rilassare a `>=` se i wheel esatti non esistono.
5. **Working tree non committato.** `README.md` risulta modificato (M) e `ROADMAP.md` è nuovo/untracked (??);
   `.serena/` e `.headroom/` sono directory generate da tool e non tracciate. Da decidere cosa committare /
   aggiungere a `.gitignore`. *(Questo file `analisi.md` si aggiungerà come ulteriore untracked.)*

---

## 12. Valutazione complessiva

**Progetto di ricerca di alta qualità, raro per disciplina metodologica.** Punti di forza:

- **Onestà intellettuale sistematica**: caveat espliciti, distinzione sampled/greedy, "no detected change ≠
  proof of equality", report ritrattati con motivazione, soglie pre-registrate.
- **Rigore statistico**: bootstrap appaiato + TOST, CI sempre riportati, determinismo seed-based.
- **Ingegneria pulita**: patch reversibile e instrumentation-only, validazione no-GPU dei patch cloud,
  verifica artefatti con manifest, codice tipizzato e documentato.
- **Due contributi distinti**: (a) il finding di compressione *answer-preserving ≠ trajectory-preserving*;
  (b) l'advisory hardware bf16/pre-Ampere come hazard di riproducibilità.

**Limiti riconosciuti dagli stessi autori**: singolo benchmark (math500) × singolo stile (sequential_light) ×
singolo seed; localizzazione inner/outer irrisolta; equivalenza greedy non dimostrata; compressione misurata è
*information-theoretic* (fake-quant), non banda wall-clock.

Il lavoro è una **misurazione stretta ma pulita** che la roadmap punta a generalizzare. Le criticità della §11
sono di igiene documentale, non di sostanza scientifica.

---
*Fine analisi. Generato in sola lettura; l'unico file scritto è questo.*
