# NL-codified street measures for PS1 Hood (Street View ↔ metric ENU)

**Project:** PS1 Hood — Street View → metric 3D in the Netherlands  
**Purpose:** Codified / widely-used public-space dimensions visible in Google Street View, usable as **scale / distance / perspective / (optionally) lens-distortion priors**.  
**Role in the pipeline:** The project already has **ENU metric poses from panos** and MapAnything densify with `is_metric_scale: true` (see [mapanything-densify.md](mapanything-densify.md)). These measures are a **validation gate** and **distortion residual check** — flag panos / blocks whose implied scale disagrees with ≥2 independent calibrators. **Do not replace MapAnything metric scale** or re-bake MA clouds from street measures; write `ok` / `scale_bias` / `distortion_fail` / `stale_sv`, don’t average calibrators into the product cloud.  
**Compiled:** 2026-09-13. Numbers from CROW, RVV 1990, BABW, NEN, RDW, EU, NSVV/NPR, municipal practice.  
**Also:** [compare-and-pathforward.md](compare-and-pathforward.md) §12.  
**Status labels used throughout:**
- **law** — RVV / BABW / Regeling voertuigen / Postregeling / Regeling verkeerslichten / Regeling kentekens
- **standard** — NEN, NPR (binding when a regulation cites them)
- **guideline** — CROW / NSVV / Fietsberaad (widely applied, municipal variance)
- **typical** — manufacturer / municipal practice, not a legal size

---

## How to use for PS1 Hood

### What this is / is not

| Do | Do not |
|---|---|
| Detect **kentekenplaat / zebra / haaientanden / Type-I bord / 30×30 tegel** in SV | Treat MapAnything / SfM scale as ground truth without a gate |
| Solve **PnP** of known-size rectangles against ENU camera poses | Invent cadastral building boxes from door heights |
| Use **lane / zebra / parking-bay lines** as vanishing + metric baseline | Assume every 30 km/h street has white centreline (CROW: ETW-bibeko **no** white lengtemarkering in the ideal profile) |
| Use **sign bottom ≥ 2.20 m** and **kenteken 520×110 mm** as vertical/scale locks | Assume lighting-mast height is a national constant |
| Flag panos whose implied scale disagrees with ≥2 independent calibrators | Trust a single parked-car length (model mix is wide) |
| Use repeating **0.50 m zebra / 0.30 m tile** grids as **distortion residual** (straight lines should stay straight in undistorted pano) | Fit a new radial model from one zebra if crop/FOV is unknown |

### Recommended lock chain (residential NL block)

1. **Hard metric** (law / NEN, mm-accurate, many instances): rear **kentekenplaat 520×110 mm**; Type-I **rond 600 mm** / **driehoek 700 mm**; **haaientanden 0.50×0.50 m**.
2. **Planar grid** (CROW / BABW, cm-accurate): **zebra 0.50 / 0.50**, **L ≥ 4.00 m**; **30×30 cm** sidewalk tiles; **parkeervak** 2.00×6.00 m (langs, guideline).
3. **Vertical** (BABW minima, not exact): sign underside **≥ 2.20 m** (bibeko, on pad/trottoir); VRI lantaarn underside **2.20–2.40 m**; curb face **~0.10–0.12 m** typical.
4. **Loose prior**: passenger-car length **mean 4.19 m / P95 4.88 m** (CROW); lighting mast **4 or 6 m** residential (municipal).
5. **Gate:** if (2) and (1) disagree by more than ~3–5% on the same pano, suspect **SV crop, wrong FOV, rolling-shutter warp, or stale capture date** — do not average them into MapAnything.

### SV capture caveats (all items)

- Google Street View car camera is **~2.5–3.0 m AGL** (typical; not a Google-published spec). Newer rigs and backpack/trekker captures differ.
- Panos are **stitched multi-camera** (historically R5/R7-class rosettes). Residual stitch seams and **rolling-shutter** warp exist near the vehicle.
- **Blurring** of faces and **kentekenplaten** is applied — plate *extent* is usually still measurable; glyphs are not.
- Capture **date** vs CROW revision: markings and signs are replaced on municipal cycles; a 2018 pano may predate a 2022 fietspad widening.
- **ETW 30 km/h / erf:** often **no** white longitudinal marking. Calibrate from plates, tiles, signs, parked cars, zebra at the junction — not from a centreline that is not there.

### Calibration-use codes (per item below)

`scale` · `baseline` · `vertical` · `vanishing` · `distortion`

---

## 1. Road markings (belijning / wegmarkering)

### 1.1 Longitudinal lines — as-, deel-, kantstreep (centre / lane / edge line)

| | |
|---|---|
| **NL / EN** | lengtemarkering: as-/scheidingsstreep, deelstreep, kantstreep / longitudinal centre, lane, edge line |
| **Width (urban)** | **GOW 50 km/h bibeko: 0.10 m** for scheidings-, deel- **and** kantstreep. GOW 70 km/h: scheiding/kant **0.15 m**, deel **0.10 m**. ETW-bibeko: **n.v.t.** (no white lengtemarkering in the ideal CROW profile). |
| **Width (rural)** | GOW: 0.15 m. Stroomweg kantstreep **0.20 m** (spitsstrook exception **0.05 m**). ETW type I kant **0.10 m**. |
| **Legal minimum** | BABW: stripe width **≥ 0.10 m** (kantstreep 0.05 m only for spitsstrook edge). Through line ≥ **20 m**. |
| **Dash/gap (guideline)** | Broken line: **1.00 m mark + 3.00 m gap** (1-3), laid on a **12 m** module. Warning line: **3.00 m + 1.00 m** (3-1). Deelstreep often **3-9**. GOW with agricultural traffic: **9-3** (0.15). Bike-lane broken edge: **1.00 + 1.00**. Double solid: EU gap **≤ 0.18 m**. |
| **Codified** | **law:** Uitvoeringsvoorschriften BABW inzake verkeerstekens, hfdst. IV §1 (min. 0.10 m); RVV 1990 art. 76. **guideline:** CROW *Richtlijnen voor de bebakening en markering van wegen 2024* §2.1, §3.2 tabel 3.1; ASVV 2021. |
| **SV visibility** | **Excellent** on GOW 50; **rare / absent** on 30 km/h erftoegangsweg. |
| **Calibration** | `scale` (0.10 m width is small — noisy at range) · `baseline` (dash period 4.00 m for 1-3; 12.00 m module) · `vanishing` · `distortion` |
| **Caveats** | Width is **function of road class**, not a universal 10 cm. Thermoplastic vs paint; wear. Dash pattern is CROW **guideline**, not BABW law except min. width. Do not assume 1-3 on every urban street. |

**CROW 2024 tabel 3.1 (excerpt, metres)**

| Category | Deelstreep | Scheidingsstreep | Kantstreep |
|---|---|---|---|
| Stroomweg nationaal (buiten kom) | 0.15 | n.v.t. | 0.20 (spitsstrook 0.05) |
| GOW buiten kom | 0.15 | 0.15 | 0.15 |
| ETW type I buiten kom | n.v.t. | n.v.t. | 0.10 |
| GOW 70 km/h bibeko | 0.10 | 0.15 | 0.15 |
| GOW 50 km/h bibeko | 0.10 | 0.10 | 0.10 |
| ETW bibeko | n.v.t. | n.v.t. | n.v.t. |

Sources: [CROW 2024 §3.2](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Richtlijnen_voor_de_bebakening_en_markering_van_wegen_2024/3.2_De_breedte_van_scheidings-,_deel-_en_kantstrepen/29287), [CROW 2024 §2.1](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/Richtlijnen_voor_de_bebakening_en_markering_van_wegen_2024/2.1_Markeringen_in_lengterichting/11069), [BABW](https://wetten.overheid.nl/BWBR0009104/2023-10-01).

### 1.2 Zebrapad (pedestrian crossing / zebra)

| | |
|---|---|
| **NL / EN** | voetgangersoversteekplaats / zebrapad — zebra (pedestrian crossing) |
| **Dimensions** | Crossing **L ≥ 4.00 m** (along the road, i.e. walkable width). White bars **0.50 m** wide, gaps **0.50 m** (CROW “vaste breedtemaat”). Bars run **parallel to the road axis**. |
| **Law vs guideline** | **law (BABW):** zebra is a marking **≥ 4 m** wide with white bars **0.4–0.6 m** wide **and** 0.4–0.6 m gap. **guideline (CROW 2024 / Ontwerpwijzer voetgangers V22):** **a1 = 0.50 m**, **a2 ≈ a1**, **L ≥ 4.00 m**. Practice in NL is almost always **0.50 / 0.50**. |
| **Codified** | BABW (zebra definition); CROW 2024 §10.1; CROW Ontwerpwijzer voetgangers V22. Bord **L2** required except at VRI. |
| **SV visibility** | **Excellent** — high-contrast, planar, repeating. Common on GOW 50; less common inside 30 km/h zones (but present at busy crossings / plateaus). |
| **Calibration** | `scale` (0.50 m period — **highest-ROI planar grid**) · `baseline` (L and count of bars → road width) · `vanishing` (bar edges ‖ road axis; stop-line ⊥) · `distortion` (period should be constant in ground plane) |
| **Caveats** | Bar edges **may follow paving joints** (not a perfect rectangle). On a plateau, talud stripes sit ≥ 5 m away (CROW). Count bars; don’t assume L = 4.00 m — that is a **minimum**. Older BABW 0.4–0.6 band still legally valid. |

Sources: [CROW 2024 §10.1](https://kennisbank.crow.nl/public/gastgebruiker/FVV/Richtlijnen_voor_de_bebakening_en_markering_van_wegen_2024/10.1_Oversteek_in_de_voorrang/29307), [CROW V22](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/Ontwerpwijzer_voetgangers/V22_Voetgangersoversteekplaats_%E2%80%93_zebrapad/118164), [BABW](https://wetten.overheid.nl/BWBR0009104/2023-10-01).

### 1.3 Stopstreep (stop line)

| | |
|---|---|
| **NL / EN** | stopstreep — stop line |
| **Dimensions** | **≥ 0.20 m** wide (BABW, RVV art. 79). At bord **B7 STOP: ≥ 0.30 m**. Bicycle stop line **0.20 m** (0.30 m allowed in tile paving). Spans the lane, between longitudinal marks or to asphalt edge. |
| **Single vs double** | Single at 30/50 km/h VRI; **double** at ≥ 60 km/h (CROW). Not a standalone marking — only with VRI or B7. |
| **Codified** | **law:** BABW hfdst. IV §1 (min. 0.20 m); B7 execution ≥ 0.30 m. **guideline:** CROW 2024 §2.2.2. |
| **SV visibility** | **Good** at signalised junctions and rare STOP signs. |
| **Calibration** | `scale` (width) · `vanishing` (⊥ road axis) · `baseline` (lane width if it spans the lane) |
| **Caveats** | Width is a **minimum**. Worn / offset from the true stop position. Double vs single encodes speed class, not a second metric. |

Sources: [CROW 2024 §2.2.2](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Richtlijnen_voor_de_bebakening_en_markering_van_wegen_2024/2.2.2_Stopstrepen/11071), [BABW](https://wetten.overheid.nl/BWBR0009104/2023-10-01).

### 1.4 Haaientanden (shark teeth / give-way triangles)

| | |
|---|---|
| **NL / EN** | haaientanden / driehoeksmarkering — shark teeth (give-way triangles) |
| **Dimensions** | **Base 0.50 m.** Height **0.50 m inside built-up area**, **0.70 m outside**. Gap ≈ base (**~0.50 m**). Bike/bromfietspad: **0.30 × 0.30 m** also allowed. Point faces the approaching driver (upstream). |
| **Codified** | **law:** RVV 1990 art. 1 + art. 80 (meaning); BABW (when they may stand without B6). **guideline:** CROW 2024 §2.2.1. Bord **B6** is (almost) always paired with them; B6 without teeth is not CROW-conform. |
| **SV visibility** | **Excellent** at every give-way junction — including 30 km/h side streets onto collectors. |
| **Calibration** | `scale` (0.50 m isosceles, many repeats) · `vanishing` (row ⊥ side-road axis) · `distortion` |
| **Caveats** | 0.50 vs 0.70 encodes **bibeko / buiten kom**. 0.30 m variant on fietspaden. Element paving may use a 0.25–0.50 m filler triangle. |

Source: [CROW 2024 §2.2.1](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Richtlijnen_voor_de_bebakening_en_markering_van_wegen_2024/2.2.1_Driehoeksmarkeringen/11070).

### 1.5 Fietssymbool (bike-lane pavement symbol)

| | |
|---|---|
| **NL / EN** | fietssymbool — bicycle symbol on the carriageway |
| **Dimensions** | CROW 2024 §2.4.3: **large vs small** model; large used at fietsstrook ≈ **1.80 m**. Small model **only bibeko**. Special ultra-narrow **width 0.75 m** (rechtsafvak / OFOS lead-in). Municipal handbooks (citing CROW fig. 2.29) commonly use: strook **< 1.50 m → 0.75 × 1.35 m**; strook **> 1.50 m → 1.10 × 2.00 m**. Word/number markings: height **1.60 m** (≤ 50 km/h bibeko) or **4.00 m** (higher speed); extra 0.20 m clearance each side. |
| **Placement** | Start of fietsstrook, after every paved side road, then every **50–100 m**. Fietsstrook edge: **0.10 m** (tiles 0.10–0.15 m); broken **1.00 / 1.00**. |
| **Codified** | **guideline:** CROW 2024 §2.4.3 fig. 2.29, §9.2. Legal status of a fietsstrook follows from the symbols + stripes (RVV). |
| **SV visibility** | **Good** on GOW with red fietsstrook; **rare** on 30 km/h mixed streets. |
| **Calibration** | `scale` (once model is classified) · `baseline` (repeat 50–100 m is too loose) |
| **Caveats** | Two sizes; tile-inlaid symbols only “look like” paint. **Do not use until large/small is classified.** Prefer zebra / plate over this. |

Sources: [CROW 2024 §2.4.3](https://kennisbank.crow.nl/public/gastgebruiker/FVV/Richtlijnen_voor_de_bebakening_en_markering_van_wegen_2024/2.4.3_Symbolen_en_verkeerstekens/11078), [CROW 2024 §9.2](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Richtlijnen_voor_de_bebakening_en_markering_van_wegen_2024/9.2_Fietsstrook/11090).

### 1.6 Other pavement marks (brief)

| Mark | Size | Status | SV | Use |
|---|---|---|---|---|
| Blokmarkering (block / rumble) | 1-3 pattern; width **0.30 m** (GOW) or **0.45 m** (stroomweg, with 0.20 kantstreep) | CROW guideline | good on GOW/stroomweg | `scale` / `baseline` |
| Gele onderbroken streep (no parking) | ratios **0.30:0.30**, **0.50:0.50** or **1.00:1.00**; width ≥ 0.10 m; min. 3 dashes | BABW / RVV art. 24 | good (often on curb) | `scale` (period) |
| Driehoek vooraanduiding B7 | CROW fig. 2.2, often **0.50 × 0.70** buiten kom | CROW | rare | `scale` |

---

## 2. Lane / road geometry

### 2.1 Rijstrook / rijbaan (lane / carriageway)

| | |
|---|---|
| **NL / EN** | rijstrook / rijbaan — traffic lane / carriageway |
| **GOW 50 km/h** | Lane width **s = 2.90–3.50 m** between markings (CROW ASVV 2021). > 3.50 m discouraged (overtaking). Recommended motor-lane next to fietsstrook **2.90 m**. |
| **GOW 70** | **3.25–3.50 m**. |
| **ETW 30 / erf (mixed)** | **Ideal carriageway 5.80 m**, **minimum 4.80 m** (two-way cars + two-way bikes). One-way car + two-way bike: ideal **4.40** / min **3.85**. One-way both: **3.85 / 3.40**. **No** white lengtemarkering in the ideal profile. |
| **GOW30** | Verharding ≥ **5.80 m** (two HGVs crawl-pass). Profiles ~5.80–10.70 m depending on fietsstroken vs vrijliggend. |
| **Codified** | **guideline:** ASVV 2021; CROW Handboek wegontwerp; Handreiking GOW30 (2023). Not law. |
| **SV visibility** | **Excellent** (the road is the image). Marked lanes only on GOW. |
| **Calibration** | `baseline` (weak — **range**, not a constant) · `vanishing` |
| **Caveats** | **Do not use 3.0 m as a scale lock.** Historic inner-city streets are often narrower than ASVV minima. Reconstruct from zebra bar-count or parking bays instead. |

Sources: [ASVV dwarsprofiel](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/ASVV_2021/Dwarsprofiel_(c)/113411), [ASVV ETW wegvak](https://kennisbank.crow.nl/public/WOBI/ASVV_2021/Wegvakvoorzieningen_op_erftoegangswegen/113086).

### 2.2 Fietsstrook / fietspad (bike lane / cycle track)

| | |
|---|---|
| **NL / EN** | fietsstrook (on carriageway) / vrijliggend fietspad — bike lane / cycle track |
| **Fietsstrook** | Recommended **2.25 m**, **minimum 1.70 m** (excl. marking). Outside kom: 1.70–2.20 m excl. mark; don’t combine with parking on GOW50. |
| **Vrijliggend fietspad (2022)** | **Absolute minimum 2.30 m** (two cyclists abreast) — raised from 2.00 m. Basisnetwerk: one-way **2.90 (2.30)**, two-way **3.60 (2.70)**. Hoofdfietsnet: 2.90 / 4.00. Doorfietsroute: 3.60 / 4.80. Recreatief: 2.60 (2.30). Obstacle-free **0.50 m** each side. |
| **Older Ontwerpwijzer** | Intensity tables 2.00 / 2.50–3.00 / 3.50–4.00 m — **superseded** for new work; **still on the ground** in many SV dates. |
| **Codified** | **guideline:** CROW Ontwerpwijzer fietsverkeer; Fietsberaad *Aanbevelingen breedte fietspaden 2022*; CROW 2024 §9.2. |
| **SV visibility** | **Excellent** (red asphalt is a NL signature). |
| **Calibration** | `baseline` (weak — width is a **range** and many existing paths are sub-standard; Fietsberaad: ~60% of in-kom paths fail the 2022 minima) |
| **Caveats** | Use as a **consistency check** against BGT/NWB, not as a scale lock. |

Sources: [CROW fietsstroken](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Ontwerpwijzer_fietsverkeer/Fietsstroken/32965), [Fietsberaad 2022](https://fietsberaad.nl/Kennisbank/Aanbevelingen-breedte-fietspaden-2022).

### 2.3 Parkeervak (parking bay)

| | |
|---|---|
| **NL / EN** | parkeervak / parkeerplaats — parking bay |
| **Personsauto, haaks / gestoken** | Width **2.50 m**, length **~5.0 m** (ASVV 2021 also cites **5.13 m** before rounding; NEN 2443 for garages). Manoeuvre aisle **~6.0 m**. |
| **Langsparkeren** | Width **2.00 m**, length **6.00–7.00 m** (often 6.00 m). |
| **Disabled** | **3.50 m** wide, or **3.00 m** + free uitstapstrook. |
| **Motorcycle** | 1.50 × 2.50 m (haaks). Bus/truck haaks: 4.00 × 13.00 m. |
| **Codified** | **guideline:** ASVV 2021 §10.8; NEN 2443 (garages only). Municipal parkeernormen copy ASVV. |
| **SV visibility** | **Excellent** on streets with marked bays; many 30 km/h streets park unmarked on-carriageway. |
| **Calibration** | `scale` / `baseline` (good **if paint is visible**; 2.00 × 6.00 langs is the residential default) |
| **Caveats** | Length 6 vs 7 m is a design choice. End bays often longer. Do not confuse with the parked **car**. |

Source: [ASVV parkeervoorzieningen](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/ASVV_2021/Maatvoering_van_parkeervoorzieningen/113583).

### 2.4 Trottoir / tegels (sidewalk / paving tiles)

| | |
|---|---|
| **NL / EN** | trottoir / stoeptegel / betontegel — sidewalk / paving slab |
| **Tile** | **Typical 300 × 300 mm** (11 tiles/m² including joints). Thickness 45–80 mm (**60 mm** common). Also **200 × 200** (heavy uitrit), **150 × 300**, **400 × 400**, and keiformaat klinkers **~210 × 70 / 210 × 105 mm**. |
| **Clear walking width** | Accessibility **min ≥ 1.50 m**, preferred **≥ 1.80 m**. Ontwerpwijzer voetgangers basisnetwerk **≥ 2.00 m** free; hoofdnet **≥ 2.90 m**. Local pinch: 0.90 m over ≤ 0.5 m, 1.20 m over ≤ 20 m. ASVV profiles often show voetpad **≥ 2.00 m**. |
| **Codified** | Tile size: **typical** (NEN-EN 1339 product standard exists; 30×30 is industry default, not a placement law). Width: **guideline** CROW Ontwerpwijzer voetgangers / Leidraad toegankelijkheid / ASVV. |
| **SV visibility** | **Excellent** — the repeating 0.30 m grid is visible even when worn. |
| **Calibration** | `scale` (count tiles; **joint included** → slightly > 0.30 m pitch, typically ~0.305 m) · `vanishing` · `distortion` (grid lines) |
| **Caveats** | Joints, cuts, mixed formats, historic brick. **Count ≥ 8 tiles** before trusting pitch. Amsterdam/older cores often use smaller klinkers. |

Sources: manufacturer catalogues (Struyk Verwo 30×30); [CROW vrije doorloopruimte](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/Ontwerpwijzer_voetgangers/Maatvoering_vrije_doorloopruimte/118122).

---

## 3. Curbs / height differences (stoeprand / trottoirband)

| | |
|---|---|
| **NL / EN** | trottoirband / stoeprand / boordsteen — kerb / curb |
| **Product sizes** | Most common bibeko bands: **13/15** (130 mm face / 150 mm back, typically 200 mm high unit) and **18/20**. RWS-band 11/22 at 160/200/250 mm (rural, 45°). Unit length typically **1000 mm**. 20 mm top chamfer. |
| **Exposed height** | **Typical 0.10–0.12 m** above carriageway (manufacturer + municipal GWW practice). Not a CROW single number for “the Dutch curb”. |
| **Dropped / inrit** | Inritblokken slope **≤ 1:6**. Carriageway-to-block step **< 0.02 m**. Uitritconstructie height **0.04–0.12 m** (zero-height not desired). Crossing afrit: width **≥ 1.20 m**, slope **≤ 1:10**, residual step **≤ 0.02 m** (municipal LIOR / CROW V2). |
| **Bus perronband** | **0.18 m** above carriageway (**guideline, treated as a hard uniformity requirement** in CROW toegankelijkheid — kneeling bus floor ~0.23 m → 0.05 m gap). Raised length 8 m of a 12 m stop. |
| **Codified** | Product: **typical** (RAW/NEN-EN 1340). Heights: **guideline** CROW *Richtlijn drempels, plateaus en uitritten*; CROW *Richtlijn toegankelijkheid*. CROW is **not** statutory (case law). |
| **SV visibility** | **Good** in oblique SV (shadow line); **poor** as a metric from a single pano (small Δh, occlusion). |
| **Calibration** | `vertical` (weak) · `baseline` (1000 mm unit length along the street, if joints visible) |
| **Caveats** | Historic cities: 14–34 mm irregularities are normal. Do **not** use 12 cm as a tight vertical lock. Bus 0.18 m is the one **repeatable** curb-height calibrator if a halte is in view. |

Sources: [Struyk Verwo trottoirbanden](https://struykverwoinfra.nl/trottoirbanden.html), [CROW uitrit ontwerp](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Richtlijn_drempels%2C_plateaus_en_uitritten/Ontwerp/27229), [CROW toegankelijke haltes](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/Richtlijn_toegankelijkheid/Toegankelijke_haltes/24161).

---

## 4. Street lighting (openbare verlichting)

| | |
|---|---|
| **NL / EN** | lichtmast / lantaarnpaal — lighting column / street-light mast |
| **Governing docs** | **NPR 13201** (NEN/NSVV, 2017/A2018) replaced **ROVL-2011**. Classes M/C/P from EN 13201 / CIE 115. **No national table of mast height × spacing.** Height and spacing are an **output of a lighting calculation** for the chosen class. |
| **Typical heights (municipal handbooks)** | **Woonstraat / erf: 4 m paaltop or 6 m + 750 mm uithouder.** Wijk-/stroomweg: **8 m + 1250 mm**. Buiten kom high intensity: **8 or 10 m**. Some municipalities are **standardising residential to 6 m** and retiring 4 m. Horst a/d Maas stock (example): most columns in **0–4.75 m** and **4.75–6.5 m** bins. |
| **Typical spacing** | **Not prescribed.** Residential pilots: **~37 m** at full NPR class, **44 m** at ~75% uniformity, **48–50 m** at ~50% (Utrechtse Heuvelrug, 6 m masts). Older sodium layouts often **~25–35 m**. |
| **Codified** | **guideline:** NPR 13201; municipal OVL beleidsplan. |
| **SV visibility** | **Excellent** (vertical repeating objects). |
| **Calibration** | `vertical` (4/6/8/10 m **classes**, not a continuous prior) · `baseline` (spacing only after clustering a whole block) |
| **Caveats** | **High municipal variance.** Decorative historic lanterns, wall-mounted gevelarmaturen, and mixed leftover stock are common. Use as a **cluster prior** after detecting many masts, never as a single-object scale. SV date vs LED replacement programmes (2015–2026) changes both height and spacing. |

Sources: NPR 13201 (NSVV/NEN); [gemeente Buren Handboek OVL 2018](https://geo.buren.nl/Woo-verzoeken/210707%20openbare%20verlichting/Bijlagen/vraag%203%20en%204/D.55677%20Documenten%20aanbesteding%20Handboek%20Openbare%20Verlichting%202018-02%20(20180725)%20actueel.pdf); municipal OVL notes (Heuvelrug, Horst a/d Maas).

---

## 5. Traffic signs (verkeersborden)

### 5.1 Plate sizes — NEN 3381 / BABW / RVV

**law:** BABW art. 16: plates (with listed exceptions) at least the sizes in **NEN 3381 §4**. Type follows speed:

| Type | Typical Vmax | Rond Ø | Vierkant side | Driehoek side | Achthoek height | Rechthoek W×H | Zonebord W×H |
|---|---|---|---|---|---|---|---|
| **0** | repeaters; D2/D3 on yellow koker | **400** | 400 | 500 | — | 300×450 | 200×250 |
| **I** | **50 / 30 km/h** (residential default) | **600** | 600 | **700** | 700 | 400×600 | 530×670 |
| **II** | 80 / 70 / 60 | 800 | 800 | 900 | 900 | 600×900 | 800×1000 |
| **III** | 120 / 130 / 100 | 1000 | 1000 | 1100 | — | 800×1200 | — |

All mm. Source: NEN 3381 bijlage A tabel A.1, reproduced in CROW *Verkeerstekens* and WIU 2020.

**Residential NL:** Type **I** dominates: round **600 mm** (A1 30, C-series), triangle **700 mm** (B6, J-warning). Type 0 (400 mm) for B1 repeaters and D2/D3 on yellow bollard.

Exceptions (not Type I): C23 spitsstrook 1850×≥1250 mm; G13/G14, K-series, several L-borden have their own sizes.

### 5.2 Mounting height and offset

| Rule | Value | Status |
|---|---|---|
| Underside of plate vs carriageway, **bibeko** | **≥ 2.20 m** | **law** BABW art. 12.A |
| Same, if on a traffic island or **off** pad/trottoir | **≥ 1.20 m** | law |
| **Buiten kom** | **≥ 1.20 m** | law |
| D2/D3 on yellow zuil | **≥ 0.90 m** | law art. 12a |
| Overhead free height, carriageway | **≥ 4.50 m** | law art. 13 |
| Overhead, fiets-/voetpad | **≥ 2.50 m** | law |
| Offset, plate edge to carriageway | preferably **0.60–3.60 m**; buiten kom without parking/vlucht **≥ 1.80 m** | law art. 14 (preferred band) |
| Plates per pole | max **two** beside or above each other (buiten kom) | law |

These are **minima**. Installed height is typically just above 2.20 m on a pad so cyclists/pedestrians clear the plate.

### 5.3 Pole (flespaal)

| | |
|---|---|
| **NL / EN** | flespaal — bottle-shaped sign post |
| **Dimensions** | **Typical Ø 76 mm lower / Ø 48 mm upper** (industry-universal NL). Straight tubes Ø 48 / 60 / 76 mm also used. Embedment ~0.60–0.80 m. Stock lengths 2.00–4.70 m chosen so plate underside meets BABW. |
| **Codified** | **typical** (VNVF / industry); height of the **plate** is law, pole diameter is not. |
| **SV visibility** | **Excellent**. |
| **Calibration** | `scale` (plate: **hard**) · `vertical` (underside ≥ 2.20 m: **one-sided**) · `scale` (Ø 48/76: weak) |
| **Caveats** | Two plates stacked → underside of the **lower** plate is the 2.20 m constraint. Street-name signs (niet-RVV) often sit lower. Type 0 on a yellow koker is 400 mm and low. |

Sources: [BABW](https://wetten.overheid.nl/BWBR0009104/2023-10-01), [CROW maatvoering borden](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/Verkeerstekens_(toepassing%2C_plaatsing_en_uitvoering)/Maatvoering/29540), [VNVF maattabellen](https://onlinebordenboek.nl/maattabellen).

---

## 6. Traffic signals / bollards / barriers

### 6.1 Verkeerslichten (traffic signals)

| | |
|---|---|
| **NL / EN** | verkeerslantaarn / VRI — traffic-signal head |
| **Lens** | Three-colour vehicle: **Ø 200 mm or 300 mm (±10%)**. Direction arrows **buiten kom: 300 mm**. Cycle and pedestrian: **200 mm (±10%)**. Yellow flasher 200 or 300. |
| **Mounting (beside carriageway)** | Underside of backboard (or lantern if no backboard) **2.20–2.40 m** AGL. Lateral clearance **≥ 0.60 m** from the rideable surface. (Toeritdosering excepted.) |
| **Codified** | **law:** *Regeling verkeerslichten* (BWBR0009151), points 9, 17, 30, 74. |
| **SV visibility** | **Excellent** at signalised junctions (less common inside a pure residential 30-block). |
| **Calibration** | `scale` (200 vs 300 must be classified; 200 mm ±10% → 180–220) · `vertical` (**tight** 2.20–2.40 band — better than signs) |
| **Caveats** | Overhead gantries use the 4.50 m free-height rule, not 2.20. Cycle lanterns are the 200 mm ones on the far-right. |

Source: [Regeling verkeerslichten](https://wetten.overheid.nl/BWBR0009151/2019-07-01).

### 6.2 Amsterdammertje / paaltje / afzetpaal (bollard)

| | |
|---|---|
| **NL / EN** | amsterdammertje / trottoirpaal / afzetpaal — bollard |
| **Classic Amsterdam steel** | Total length **1.06 m**, **~0.75 m AGL**, ~0.30–0.35 m buried. Conical, ~110–168 mm diameter. RAL 3007. **Typical / historic Amsterdam spec**, not a national CROW size. Being removed in many streets. |
| **Concrete “amsterdammertje” clones** | Often **1.05 m including 0.30³ m foot** → ~0.75 m AGL. |
| **Flexible PU urban bollard** | **Typical Ø 80 × 700–800 mm AGL** (UN 12899-3 class products). Red/white or anthracite + retro bands. **Not CROW-mandated.** |
| **SV visibility** | **Good** where still present; Amsterdam much reduced vs 1990s. Other cities use mixed plastic/steel. |
| **Calibration** | `vertical` (0.75 m is a **useful loose prior** if the object is classified as this family) · `scale` (Ø 80 mm flexi) |
| **Caveats** | Do not treat “paaltje = 75 cm” nationally. Classify first. |

Sources: [Wikipedia NL Amsterdammertje](https://nl.wikipedia.org/wiki/Amsterdammertje_(paaltje)) (secondary; matches supplier 750 mm AGL); supplier specs (typical).

---

## 7. Vehicles as scale

### 7.1 Kentekenplaat (NL licence plate) — **highest-ROI rigid rectangle**

| Vehicle | Size (mm) | Notes |
|---|---|---|
| Cars, vans, campers, trailers > 750 kg | **520 × 110** (landscape) or **340 × 210** (two-line) | Yellow + black border + EU blue strip. **Default rear plate is 520×110.** |
| Motorcycles | **210 × 143** | |
| Brom-/snorfiets | **145 × 125** landscape or **100 × 175** portrait | |
| Bijzondere bromfiets | **100 × 120** | |
| US-spec allowed (model 18.2) | **310 × 110** | Rare |
| Oldtimer dark-blue (pre-1978) | **445 × 105** or **275 × 195** (among others) | Rare in a random block |
| EU blue strip (Council Reg. 2411/98) | strip height **≥ 98 mm**, width **40–50 mm**; 12 stars on 15 mm radius; country code **≥ 20 mm** | Same on 520×110 NL plates |

**Codified (law):** *Regeling kentekens en kentekenplaten* BWBR0009071 (models in the annex); RDW execution. EU 2411/98 specifies the **blue strip**, not the 520×110 overall size — that is **NL law** aligning with the common European format.

**SV:** **Excellent** count (almost every parked car); Google **blurs glyphs** but the yellow rectangle remains. Rear plates face the camera on parked langs cars more often than fronts.

**Calibration:** `scale` (mm-accurate) · `PnP` (4 corners) · `distortion` (aspect 520/110 = **4.727** is a strong residual) · `vertical` (weak: mounting height not NL-fixed; typically bumper ~0.30–0.50 m AGL).

**Caveats:** Two-line **340×210** on 4×4 / US imports / some vans. Bike carriers show a **white** 520×110 repeat. Plastic surrounds add 5–15 mm — measure the **yellow field + black border**, not the holder. Blur can inflate the box by a few px — detect the black border.

Sources: [RDW soorten kentekenplaten](https://www.rdw.nl/de-kentekenplaat/soorten-kentekenplaten), [BWBR0009071](https://wetten.overheid.nl/BWBR0009071/2025-07-01), [EU 2411/98](https://eur-lex.europa.eu/eli/reg/1998/2411/oj/eng).

### 7.2 Passenger cars (personenauto)

CROW *Karakteristieken van voertuigen* (width **excl. mirrors**):

| | Mean | P5 | P95 | Legal max (Regeling voertuigen) |
|---|---|---|---|---|
| Length | **4.19 m** | 3.50 | **4.88 m** | 12.00 m |
| Width | **1.70 m** | 1.57 | **1.83 m** | 2.55 m |
| Height | **1.63 m** | 1.51 | **1.73 m** | 4.00 m |
| Wheelbase | 2.54 m | 2.31 | 2.77 m | — |

**Normvoertuig (CROW):** 4.88 × 1.83 × 1.73 m, wheelbase 2.77 m. Ground clearance ~0.10 m.

EU type-approval / weights & dimensions: max width **2.55 m** (Commission IR 2021/535 / Dir. 96/53/EC family) — a **ceiling**, not a typical car.

**SV:** Excellent count. **Calibration:** `scale` **loose prior only** (σ_length ≈ 0.42 m). Prefer the **plate**. Mirrors are excluded from CROW width; they **are** in the image.

Source: [CROW standaardpersonenauto](https://kennisbank.crow.nl/public/gastgebruiker/FVV/Karakteristieken_van_voertuigen_en_mensen/Standaardpersonenauto/14993).

### 7.3 Bus / truck (when in frame)

| Class | Typical / legal | Use |
|---|---|---|
| Standaard stadsbus | CROW design **L 12.10 m (P95)**, W 2.47, H 3.10; legal 2-axle **≤ 13.50 m**, 3-axle **≤ 15.00**, articulated **≤ 18.75**, W **2.55**, H **4.00** | Length as loose prior if the vehicle is classified; halte 0.18 m perron is better |
| Rigid truck | legal **L ≤ 12.00**, tractor+semi **16.50**, truck+trailer **18.75**, W 2.55 (reefer 2.60), H 4.00 | Same |
| Halte geometry | inrij 24 m, perron = vehicle length, uitrij 15 m; haven depth **3.0 m** (min 2.8) | `baseline` if painted |

Sources: [CROW autobus](https://kennisbank.crow.nl/public/gastgebruiker/902/Basiscriteria/Ontwerpvoertuig_autobus/11795), [ASVV autobussen](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/ASVV_2021/Autobussen/113723), [CROW vrachtauto](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Handboek_wegontwerp_buiten_de_bebouwde_kom/Vrachtauto/116083).

---

## 8. Parking meters / trash bins / bus stops

**Not tightly standardized as object sizes.** Use only the items below; skip generic “prullenbak” and “parkeerautomaat” as scale.

| Item | What is codified | SV | Use |
|---|---|---|---|
| **Bushalte perron** | Height **0.18 m**; raised length 8 m (12 m bus); free width ≥ **1.60 m** (provincial handbooks) / CROW free 1.5 (pref 1.8); haven depth 3.0 m; abri if present must be wheelchair-accessible | good | `vertical` 0.18 m; `baseline` 8 / 12 m |
| **Parkeerautomaat** | **No size standard.** Placement guideline: visible from the bay, preferably **≤ 100 m** walk, ~1 per 50 bays (CROW Handboek parkeren). Mixed vendors (Cale, Scheidt, ParkeerService…) | good as detection, **bad as scale** | skip for metric |
| **Afvalbak** | **No national size.** At stops: outside the 0.90 m walkway, ≥ 1.2 m from perron edge | good | skip |
| **Abri** | Municipal catalogue (e.g. JCDecaux) — model-specific; not CROW-universal | good | only if the **exact municipal model** is known |

Sources: [CROW toegankelijke haltes](https://kennisbank.crow.nl/public/gastgebruiker/WOBI/Richtlijn_toegankelijkheid/Toegankelijke_haltes/24161), [CROW parkeerapparatuur](https://kennisbank.crow.nl/public/gastgebruiker/PARK/Handboek_parkeren/Betaalwijzen_(parkeerapparatuur)/14686).

---

## 9. Speed bumps / drempels / plateaus

CROW *Richtlijn drempels, plateaus en uitritten*. Heights **0.08 or 0.12 m**. Profile is **sinusoidal** except trapezium Drempel-50 at 12 cm:  
`y = (H/2) {1 − cos(2π x / L)}`.

**Recommended drempels (full width of the road)**

| Design speed | Shape | H | Oprit | Top | Afrit | **Total L** |
|---|---|---|---|---|---|---|
| 20 | sinus | 0.08 | 1.00 | — | 1.00 | **2.00 m** |
| 30 | sinus | 0.12 | 2.40 | — | 2.40 | **4.80 m** |
| 30 | sinus | 0.08 | 1.75 | — | 1.75 | **3.50 m** |
| 50 | trapezium | 0.12 | 4.80 | 2.40 | 4.80 | **12.00 m** |
| 50 | sinus | 0.08 | 3.00 | — | 3.00 | **6.00 m** |
| 60 | sinus | 0.12 | 6.00 | — | 6.00 | **12.00 m** |

**Plateaus:** top **> 2.40 m** (often much longer at junctions). 8 cm preferred for bus/HGV; 12 cm when a level pedestrian crossing is wanted. Amsterdam GOW30 zebra-on-plateau practice (municipal, 2024): sinus **12 cm**, flat **≥ 8 m**, ramps 1.50 m (no bus) / 2.40 m (bus).

White **taludmarkering** on the ramps is common (high-contrast in SV).

**Codified:** **guideline** CROW. **SV:** **Good** (asphalt colour change + talud stripes). **Calibration:** `scale` (L is the useful number; H is hard to read in SV) · `baseline`. **Caveats:** Built profile drifts (CROW gives ±2–8 mm height tolerances). Identify 30 vs 50 from context before using L.

Sources: [CROW toepassing drempels](https://kennisbank.crow.nl/public/gastgebruiker/FVV/Richtlijn_drempels%2C_plateaus_en_uitritten/Richtlijnen_voor_de_toepassing_van_verkeersdrempels_en_-plateaus/27209), [CROW hoogteprofiel](https://kennisbank.crow.nl/public/gastgebruiker/WOBU/Richtlijn_drempels%2C_plateaus_en_uitritten/Bijlage_II_Hoogteprofiel_drempel_en_plateaus/27216).

---

## 10. Building-adjacent (only if codified)

**Skip:** door heights, window rhythm, storey heights, brick bonds as “the Dutch brick” — not reliably codified for SV scale.

### 10.1 Brievenbus / letter slot

| | |
|---|---|
| **NL / EN** | brievengleuf / brievenbus — letter slot / mailbox |
| **Opening** | Free aperture **≥ 265 × 32 mm**, horizontal. |
| **Height** | **Preferred 1.10 m** AGL of the serving surface; **legal band 0.60–1.80 m**. |
| **Internal box** | ≥ 270 × 150 × 380 mm. |
| **Setback** | ≤ 10 m from the public road; serving level within ±2.5 m of carriageway (no long stair). House number on or next to the box. |
| **Codified** | **law:** Postregeling 2009 art. 6 (BWBR0025578). NEN-EN 13724 is the product test standard (apertures); NL legal sizes are the Postregeling. |
| **SV visibility** | **Rare / poor** (small, often in the door, often occluded). Outdoor boxes better. |
| **Calibration** | `scale` / `vertical` only if a **street-facing outdoor box** is clearly resolved — otherwise skip. |
| **Caveats** | Decorative slots may be non-compliant. Height band is 1.2 m wide — weak vertical. |

Sources: [Postregeling 2009 art. 6](https://wetten.overheid.nl/BWBR0025578), [PostNL folder](https://www.postnl.nl/api/assets/blt43aa441bfc1e29f2/bltc332b00ddf9cd77a/uw-brievenbus-postnl-2017/).

### 10.2 Huisnummerplaat (house-number plate)

**NEN 1774:1959** (afmetingen huisnummerbordjes) was **withdrawn 2010-05-01**. No current national size. Municipal façade rules vary (contrast, height, illumination) without a shared millimetre spec.

**Calibration: skip** as a metric prior. Use only as a **semantic** house-ID, not a ruler.

---

## Top 10 highest-ROI calibrators — residential NL block (e.g. smoke-dense)

Ranked for a **30 km/h erftoegangsweg / woonblok**: many parked cars, 30×30 tegels, Type-I signs, maybe one zebra / haaientanden at the junction, **often no centreline**.

| Rank | Calibrator | Number | Why ROI | Use | Trust |
|---|---|---|---|---|---|
| **1** | **Kentekenplaat** (rear, yellow) | **520 × 110 mm** (law) | mm-accurate, dozens per block, 4.727 aspect, PnP-ready even when blurred | `scale` `PnP` `distortion` | **Hard** |
| **2** | **Zebrapad bars** | **0.50 / 0.50 m**, L ≥ 4.00 m (CROW; BABW 0.4–0.6) | Planar grid, vanishing ‖ road, counts → road width | `scale` `vanishing` `distortion` | **Hard** (confirm 0.50 not 0.40) |
| **3** | **Haaientanden** | **0.50 × 0.50 m** bibeko (CROW) | Every give-way side street; repeating | `scale` `vanishing` | **Hard** |
| **4** | **RVV Type-I plate** | rond **Ø 600 mm** / driehoek **700 mm** (NEN 3381); underside **≥ 2.20 m** (BABW) | Known rectangle/circle above ground; pairs scale + vertical | `scale` `vertical` | **Hard** size; height is a **minimum** |
| **5** | **30×30 cm trottoirtegel** | pitch **~0.30 m** (typical) | Dense grid along the camera path | `scale` `vanishing` `distortion` | **Medium** (joints, mixed formats) — count ≥ 8 |
| **6** | **Langsparkeervak** | **2.00 × 6.00 m** (ASVV guideline) | Painted boxes on many woonerfs | `scale` `baseline` | **Medium** (6–7 m length choice) |
| **7** | **VRI lantern** (if present) | lens **Ø 200 mm ±10%**; underside **2.20–2.40 m** (law) | Tight vertical band | `vertical` `scale` | **Hard** vertical; classify 200 vs 300 |
| **8** | **Drempel 30** | L **3.50 m** (8 cm) or **4.80 m** (12 cm) (CROW) | Talud stripes visible | `baseline` | **Medium** (must pick 8 vs 12 cm class) |
| **9** | **Personenauto length** | mean **4.19 m**, P95 **4.88 m** (CROW) | Always there; high variance | `scale` **prior only** | **Loose** — never sole lock |
| **10** | **Lichtmast cluster** | **4 or 6 m** residential (municipal typical); spacing **~25–50 m** | Repeating verticals for a block-level check | `vertical` `baseline` | **Loose** — classify 4 vs 6 first |

**Honourable mentions (use when in view):** stopstreep ≥ 0.20/0.30 m; fietsstrook 0.10 m + 1-3/1-1 edge; bus perron **0.18 m**; flespaal Ø 76/48 mm; amsterdammertje **~0.75 m AGL** (Amsterdam / clones only).

**Do not use as scale:** huisnummerplaten, door heights, generic bins/meters, fietspad width, GOW “3 m lane”, lighting spacing from one pair of masts.

---

## Worked recipe: SV → ENU validation gate

1. **Detect** in each SV pano (or cubemap face): kenteken rectangles, Type-I circles/triangles, zebra bar pairs, haaientanden, tile vanishing.
2. **Undistort / uncrop check:** a 520×110 plate should project with aspect **4.727 ± ~2%** after the assumed GSV camera model. Zebra 0.50 m bars should have **constant ground-plane period**. Systematic barrel/pincushion across the pano is a **lens/stitch flag**, not a scale error.
3. **PnP** each hard rectangle into the pano’s ENU camera. Residual translation along the viewing ray is the **scale error** of that camera.
4. **Robust aggregate** per block: median of plate PnPs (n ≥ 5) is the scale gate. Zebra/teeth should agree within **~3%**. If plates say 1.00 and zebra says 1.08, **do not average** — inspect crop/FOV/date.
5. **Vertical:** Type-I underside should land **≥ 2.20 m** AGL in ENU. A cluster sitting at 1.6 m implies a **vertical scale or origin** problem (or the plates are on an island, legal ≥ 1.20 m — filter by “on pad”).
6. **Cars:** use as a **sanity band** (4.0–5.2 m covers ~P5–P99). Outliers are vans / cropped detections.
7. **Write the gate**, don’t bake it into MapAnything: `ok` / `scale_bias` / `distortion_fail` / `stale_sv`.

---

## Source index (primary)

| Doc | What it governs | URL |
|---|---|---|
| RVV 1990 | Meaning of marks/signs | wetten.overheid.nl BWBR0004825 |
| Uitvoeringsvoorschriften BABW inzake verkeerstekens | Marking minima, sign sizes/heights | https://wetten.overheid.nl/BWBR0009104/2023-10-01 |
| NEN 3381 | Traffic-sign dimensions (types 0–III) | cited by BABW; tables in CROW |
| CROW Richtlijnen bebakening en markering 2024 | Line widths, dashes, zebra, teeth, symbols | kennisbank.crow.nl (guest pages cited above) |
| CROW ASVV 2021 | Urban cross-sections, parking, ETW/GOW | kennisbank.crow.nl |
| CROW Ontwerpwijzer fietsverkeer + Fietsberaad 2022 | Bike-lane / path widths | https://fietsberaad.nl/Kennisbank/Aanbevelingen-breedte-fietspaden-2022 |
| CROW Richtlijn drempels, plateaus en uitritten | Bump/plateau/uitrit | kennisbank.crow.nl |
| CROW Ontwerpwijzer voetgangers / V22 | Zebra, sidewalk clear width | kennisbank.crow.nl |
| CROW Richtlijn toegankelijkheid | Bus stop 0.18 m, free widths | kennisbank.crow.nl |
| Regeling verkeerslichten | Lens Ø, lantern height 2.20–2.40 m | https://wetten.overheid.nl/BWBR0009151/2019-07-01 |
| Regeling kentekens en kentekenplaten | Plate millimetres | https://wetten.overheid.nl/BWBR0009071/2025-07-01 |
| RDW | Same, human-readable | https://www.rdw.nl/de-kentekenplaat/soorten-kentekenplaten |
| Council Regulation (EC) 2411/98 | EU blue strip geometry | EUR-Lex |
| Regeling voertuigen + Dir. 96/53/EC | Vehicle max L/W/H | wetten.overheid.nl |
| Postregeling 2009 art. 6 | Letter slot | https://wetten.overheid.nl/BWBR0025578 |
| NPR 13201 / NSVV | Lighting **quality classes**, not mast tables | nsvv.nl |
| Municipal OVL handbooks | 4/6/8/10 m practice | e.g. Buren 2018 |

CROW publications are **guidelines** adopted by almost every gemeente; they are not statutes. BABW/RVV/Regelingen **are**. When both exist (zebra 0.50 vs 0.4–0.6), treat CROW as the **as-built prior** and BABW as the **legal envelope**.
