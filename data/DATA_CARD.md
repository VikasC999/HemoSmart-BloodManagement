# HemoSmart synthetic patient data: data card

`data/patient_data.csv` is **synthetic**. No real patient record was used. It is made by
[`data/create_dataset.py`](create_dataset.py) (fixed seeds, 5,000 patients) and labelled by
[`data/label_process.py`](label_process.py). This card lists how each variable is drawn, where each
number comes from, how it was checked against real data, and what has no source.

Regenerate: `python data/create_dataset.py`, then `python models/train_xgboost.py`.

## Columns

| Column | Model input? | How it is generated |
|---|---|---|
| `sex` | **no** | M/F, 50/50 (assumption). Only shapes pre-op Hb. Transfusion thresholds are not sex-specific, so the model doesn't see it |
| `hemoglobin_preop` | no | Pre-operative Hb by sex: men N(14.0, 1.6), women N(12.3, 1.6) g/dL, clipped 6–18.5 |
| `hemoglobin` | yes | Hb **at the transfusion decision**: `hemoglobin_preop` minus a surgery-specific peri-operative drop (Gamma, table below), clipped 4–18.5. Rounded to 0.01 |
| `platelets` | yes | N(250k, 70k)/µL clipped 50k–450k, then an 8% thrombocytopenic subgroup redrawn uniform 10k–120k (dengue, chemotherapy, liver disease) |
| `INR` | yes | Right-skewed: 0.75 + LogNormal(ln 0.27, 0.58), clipped 0.8–4.0. Median 1.02, about 4% above 1.5 |
| `age` | yes | Uniform 18–89 |
| `surgery_type` | yes | Cardiac 20%, Orthopedic 25%, General 40%, Emergency 15% |
| `needs_prbc`, `needs_platelets`, `needs_ffp` | labels | A simulated clinician decision (D50): logistic soft boundary around the protocol threshold for that surgery (width Hb 0.3 g/dL, platelets 6,000/µL, INR 0.08), then 1% record noise |
| `transfusion_needed` | label | Any of the three |

### Peri-operative Hb drop (g/dL)

| Surgery | Drop mean (SD) | Protocol Hb threshold | PRBC rate here | Published rate | Source |
|---|---|---|---|---|---|
| Cardiac | 4.0 (1.0) | 9 | 47.7% | 52.3% restrictive, 72.6% liberal (TRICS III); hospital range 7.8–92.8% (STS CABG) | Mazer et al., NEJM 2017, https://www.tctmd.com/news/trics-iii-restrictive-red-cell-transfusion-noninferior-liberal-use-cardiac-surgery ; Bennett-Guerrero et al., JAMA 2010, https://scholars.duke.edu/publication/742147 |
| Orthopedic | 3.0 (0.8) | 8 | 14.3% | THA 22.2%, TKA 18.3% (NSQIP); 16% with tranexamic acid | https://vivo.weill.cornell.edu/display/pubid25471908 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC4383080 |
| General | 2.0 (1.0) | 8 | 8.3% | about 15% red-cell use across elective surgery, Indian tertiary centre | https://impressions.manipal.edu/open-access-archive/4217 ; https://nmji.in/?p=5786 |
| Emergency | 1.2 (1.5) | 10 | 20.9% | about 17% of 25,599 trauma patients (US level I) | https://experts.arizona.edu/en/publications/blood-transfusions-in-trauma-six-year-analysis-of-the-transfusion/ |

The drop sizes are **calibrated** so the PRBC rates land in those ranges with the protocol thresholds;
they aren't measured drops. Only the cardiac 3–4 g/dL (haemodilution on bypass plus blood loss) is
close to a typical reported figure. Emergency has a smaller mean drop because its protocol threshold
is already 10 g/dL; its wide right-skewed spread stands for the few patients who bleed heavily.

Overall rates (5,000 patients): PRBC 19.4%, platelets 4.9%, FFP 6.6%, any transfusion 28.2%.
Before this card (no sex, no drop, INR N(1.1, 0.35)): PRBC 6.1%, platelets 4.9%, FFP 15.4%, cardiac
PRBC 7.8%. FFP was then the most-ordered component, the opposite of real practice.

## Parameter sources

| Parameter | Value used | Real-world value | Source |
|---|---|---|---|
| Hb, US adults (reference) | n/a | men 14.4–15.0, women 13.0–13.3 g/dL (SD 1.1–1.4) | NHANES 2021–23 CBC_L/DEMO_L, https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.htm |
| Anaemia cut-offs | M < 13, F < 12 g/dL | same | WHO 2024, *Guideline on haemoglobin cutoffs to define anaemia*, https://www.who.int/publications/i/item/9789240088542 |
| India anaemia | men 26%, women 42% (generated) | NFHS-5 2019–21: men 15–54 25.0%, women 15–49 57.0% | https://eparlib.nic.in/bitstream/123456789/2979123/1/AU1953_osmtyG.pdf ; https://rchiips.org/nfhs |
| Pre-operative anaemia | 33.9% (generated) | 30.1% overall, 32.5% in LMICs (ISOS, 38,770 patients) | Fowler et al., Br J Anaesth 2018, https://qmro.qmul.ac.uk/xmlui/handle/123456789/49263 |
| Platelets | N(250k, 70k), clip 50–450k | 256 (68) ×10³/µL; women higher | NHANES (above); https://www.jcdr.net/ReadXMLFile.aspx?id=2630 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC3561305 |
| INR | ~4% > 1.5 | healthy about 0.8–1.2; no clean pre-op > 1.5 figure (one elective cohort: abnormal PT 9.5%) | https://www.medigraphic.com/pdfs/cirgen/cg-2018/cgi183d.pdf |
| Blood groups (`data/demand_process.py`) | O 38, B 32.5, A 22, AB 7.5, Rh− 5% | O 37.12, B 32.26, A 22.88, AB 7.74, Rh− 5.39% (10,000 donors, all-India) | Agrawal et al., Asian J Transfus Sci 2014, https://pmc.ncbi.nlm.nih.gov/articles/PMC4140055 |

The pre-op Hb means (14.0 / 12.3) are NHANES shifted down to reach roughly the NFHS-5 / ISOS anaemia
levels. **That shift is our calibration, not a published number.** Women land below NFHS-5's 57%
because NFHS-5 is a general-population survey of women 15–49; pushing women to 57% would put overall
pre-op anaemia at about 41%, well above ISOS.

## Thresholds: AABB 2023 next to the HemoSmart local protocol

The labels use HemoSmart's **local protocol** ([schema/clinical_thresholds.py](../schema/clinical_thresholds.py),
D52), not a published guideline. They are kept on purpose (the RAG explanation, doses and tests use the
same numbers); the differences are listed so nobody mistakes them for AABB numbers.

| Criterion | HemoSmart protocol | AABB / ICTMG guideline | Source |
|---|---|---|---|
| RBC, stable adult / elective | Hb < 8 g/dL | Hb < 7 (8 for orthopaedic or pre-existing CVD) | Carson et al., AABB 2023, JAMA 2023;330(19):1892–1902, https://www.isbtweb.org/resource/red-blood-cell-transfusion-2023-aabb-international-guidelines.html |
| RBC, cardiac surgery | Hb < 9 | Hb < 7.5 | same |
| RBC, emergency | Hb < 10 | no Hb cut-off; haemorrhage is managed clinically | same |
| Platelets, major surgery | < 50k/µL | 50k for major non-neuraxial surgery | Metcalf et al., AABB/ICTMG 2025, https://www.aabb.org/news-resources/news/article/2025/05/29/aabb-develops-new-platelet-transfusion-guidelines |
| Platelets, cardiac surgery | < 100k/µL | **no guideline found** giving 100k for cardiac surgery | none |
| FFP | INR > 1.5 | used as the appropriateness benchmark in Indian FFP audits; WHO handbook: INR < 2.0 before most surgery | https://www.msjonline.org/index.php/ijrms/article/view/4130 |

## Realism checks

- **Patients vs NHANES 2021–23** (survey-weighted, adults): [evaluation/results/data_realism.md](../evaluation/results/data_realism.md),
  `python -m evaluation.data_realism`. Pre-op Hb is about 0.85 g/dL below US adults in both sexes,
  with a wider spread (SD 1.6 vs 1.2) and 26% / 42% anaemia vs 5.6% / 13.2%: intended, since the target
  is Indian surgical patients. The upper tail (95th percentile) matches within 0.1 g/dL. Platelets:
  the 95th percentile matches (363k vs 375k); the lower tail is heavier by design (15% below 150k vs
  2.5%), from the thrombocytopenic subgroup. Blood groups match Agrawal 2014 to within 0.9 points.
- **Donors vs real donors** (UCI Blood Transfusion Service Center): [evaluation/results/donor_real_data.md](../evaluation/results/donor_real_data.md).

## Assumptions without a published source

- Mean pre-operative Hb in Indian surgical patients (no national study found); the 14.0 / 12.3 means are calibrated.
- Share of pre-operative patients with INR > 1.5 (the ~4% is an estimate).
- Hospital share of thrombocytopenic patients (the 8% subgroup).
- The 50/50 sex mix of surgical patients, and the same mix for every surgery type.
- The size of each peri-operative Hb drop (calibrated to the published transfusion rates above, not measured).
- An official NBTC/DGHS adult clinical-use guideline with platelet/FFP thresholds (not found).
- The 100k cardiac platelet threshold (no guideline found).

## Known limitations

- One generated cohort, not a real hospital. The labels are simulated decisions, so the model can at
  best recover the simulator (the benchmark's ceiling); it says nothing about accuracy on real patients.
- Variables are independent apart from sex → Hb and surgery → Hb drop. Real patients have correlations
  the generator lacks (age and anaemia, cardiac patients on anticoagulants raising INR, cardiac bypass
  lowering platelets, more men in cardiac surgery).
- Platelets aren't sex-specific (NHANES women run ~30k higher) and nothing models a peri-operative
  platelet or INR change.
- Age is uniform 18–89, not a real surgical age mix.
- Only Hb and platelet marginals are checked against real data. INR and transfusion rates by surgery are
  calibrated to published figures, not to patient-level data. MIMIC-IV Demo (100 US ICU patients) is
  the only open patient-level transfusion source found and is not used.
