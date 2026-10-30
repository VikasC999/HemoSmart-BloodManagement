# HemoSmart synthetic data: real-world sources (research, 2026-10-30)

Scope: each generator parameter vs a citable real value. "Verified" = I opened the page or file myself. "Excerpt" = the figure comes from a search abstract/snippet and the full text was not opened. A blank source means I could not find one. Nothing here is invented: where there is no source, the table says so.

How the generator behaves today (I recomputed it read-only with the same seeds, 5,000 patients):
any transfusion 24.1%; PRBC 6.1%, Platelets 4.9%, **FFP 15.4%**; by surgery, PRBC: Cardiac 7.8%, Ortho 3.7%, General 3.6%, Emergency 15.0%. P(INR>1.5)=13.0%, P(Hb<8)=2.1%.

---

## 1. ABO/RhD distribution (India)

| Parameter | Current (file:line) | Real-world value | Source |
|---|---|---|---|
| ABO shares (summed over Rh) | `data/demand_process.py:15-18`: O 38.0, B 32.5, A 22.0, AB 7.5 | **All-India, 10,000 donors, 5 regions, 2009-11:** O 37.12, B 32.26, A 22.88, AB 7.74. B leads only in the North | Agrawal et al., Asian J Transfus Sci 2014;8(2):121-5. https://pmc.ncbi.nlm.nih.gov/articles/PMC4140055 (excerpt) |
| Rh(D)-negative share | same: 5.0% | All-India 5.39% (94.61% Rh+) | same |
| Bengaluru (SJBIT's city) | n/a | 36,964 donors: O 39.81, B 29.95, A 23.85, AB 6.37, Rh- 5.79% | Periyavan et al., Asian J Transfus Sci 2010. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2847344/ (excerpt) |
| Regional contrast | n/a | W. Rajasthan (83,631): B 36.4, O 31.7, A 22.2, AB 9.4, Rh- 8.25%. S. Gujarat (40,732): B 34.4, O 32.3, A 24.4, AB 8.9 | https://njmr.in/index.php/file/article/view/580 (excerpt) |
| **Donor pool blood types** | `agents/donor_module.py:91` `rng.choice(BLOOD_GROUPS)`: **uniform, 12.5% each** (so O- is 12.5% of donors) | Same as above: O- about 2%, AB- about 0.3-0.5% | Agrawal 2014 (above) |

**Recommendation:** keep `GROUP_SHARE` (it matches Agrawal 2014 to within 1 point). **Change** `simulate_donors` to draw blood type with `GROUP_SHARE` weights: today the donor pool has 6x too many O- donors, which makes O- shortages easy to fix.

## 2. Hb, platelets, INR distributions

### 2a. General adult population (NHANES 2021-23, computed by me, unweighted, age 18+, n=6,004)

Files (verified, downloaded): `CBC_L.xpt` https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt (doc: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.htm), demographics `DEMO_L.xpt` (same folder). The earlier cycle is `CBC_J.xpt` / `DEMO_J.xpt` under `/Public/2017/DataFiles/` (URL verified, returns 200). Variables: **LBXHGB** = haemoglobin (g/dL), **LBXPLTSI** = platelets (1000 cells/uL), join on **SEQN**, sex **RIAGENDR** (1=M, 2=F), age **RIDAGEYR**. NHANES CBC has **no INR/PT**.

| Group | Hb mean (SD) g/dL | Platelets mean (SD) x10^3/uL |
|---|---|---|
| M 18-39 / 40-59 / 60+ | 15.04 (1.07) / 14.91 (1.32) / 14.40 (1.41) | 249.5 (54.9) / 243.1 (57.8) / 227.1 (62.6) |
| F 18-39 / 40-59 / 60+ | 13.01 (1.22) / 13.23 (1.30) / 13.31 (1.20) | 285.3 (69.0) / 276.8 (70.4) / 257.7 (69.3) |
| All adults | 13.89 (1.48) | 256.0 (68.1) |
| 2.5-97.5th pct | M 11.6-17.1; F 10.4-15.3 | M 141-374; F 158-432 |
| Share below cut-off | M Hb<13: 8.1%; F Hb<12: 13.2%; Hb<8: 0.1-0.2% | <150k: M 4.0%, F 1.8% |

### 2b. India and surgical patients

| Parameter | Current (file:line) | Real-world value | Source |
|---|---|---|---|
| Hb distribution | `data/create_dataset.py:41` N(12.5, 2.2) clipped 5-16, **no sex variable** | US adults 13.9 (1.5); men about 2 g/dL higher than women (table above) | NHANES CBC_L (computed above) |
| Anaemia cut-offs | n/a | Men <13.0, non-pregnant women <12.0 g/dL (kept in the 2024 guideline) | WHO 2024, *Guideline on haemoglobin cutoffs to define anaemia*. https://www.who.int/publications/i/item/9789240088542 (page verified; cut-offs from excerpt) |
| India anaemia prevalence | generator: P(Hb<12)=40%, P(Hb<13)=58% (sexless) | NFHS-5 2019-21: women 15-49 **57.0%**, men 15-54 **25.0%** | NFHS-5 via Lok Sabha annexure https://eparlib.nic.in/bitstream/123456789/2979123/1/AU1953_osmtyG.pdf (excerpt); report: https://rchiips.org/nfhs |
| Pre-op anaemia (surgical) | n/a | ISOS, 38,770 elective inpatients, 27 countries: **30.1%** anaemic; LMIC 32.5%, HIC 28.5% | Fowler et al., Br J Anaesth 2018;121(6):1227-35. https://qmro.qmul.ac.uk/xmlui/handle/123456789/49263 (excerpt) |
| Platelets | `create_dataset.py:42` N(250k, 70k) clipped 50k-400k | 256 (68) x10^3; women higher. Chennai healthy adults: F 261.6, M 242.3 x10^9/L. Italy 40,987: men 141-362, women 156-405 | NHANES; JCDR 2012 https://www.jcdr.net/ReadXMLFile.aspx?id=2630 (excerpt); Biino et al., PLoS One 2013 https://pmc.ncbi.nlm.nih.gov/articles/PMC3561305 (excerpt) |
| Thrombocytopenic subgroup | `create_dataset.py:58-61`: 8%, uniform 10k-120k | **No solid source** for an 8% hospital share. General population <150k is only 2-4% (NHANES) | none found |
| INR | `create_dataset.py:43` N(1.1, 0.35), so **13% have INR>1.5** | Healthy range about 0.8-1.2. Pre-op INR>1.5 prevalence: **no clean figure found**. Mexican elective abdominal cohort (n=263): 1.1% in the highest-risk INR band, abnormal PT 9.5% | https://www.medigraphic.com/pdfs/cirgen/cg-2018/cgi183d.pdf (excerpt; band cut-offs not stated) |
| Age | `create_dataset.py:44` uniform 18-89 | Not researched in depth; ISOS LMIC patients were younger than HIC | Fowler 2018 |

**Recommendation:** add `sex` (about 50/50) and draw Hb by sex: men around N(14.0, 1.6), women around N(12.3, 1.6). That is NHANES shifted down to give roughly NFHS-5 / ISOS anaemia levels; the shift is my calibration, not a published number. Keep platelets (raise the clip to about 450k). **Change INR** to a right-skewed draw, e.g. lognormal with a median near 1.0, so about 3-5% exceed 1.5. Today 13% do, and that makes FFP (15.4%) the most-ordered component, which is the opposite of real practice.

## 3. Transfusion rates by surgery and guideline thresholds

| Parameter | Current (file:line) | Real-world value | Source |
|---|---|---|---|
| Cardiac: % given RBC | generator 7.8% | TRICS III (5,243 moderate/high-risk cardiac surgery pts): **52.3%** restrictive (<7.5) vs **72.6%** liberal (<9.5); median 2 vs 3 units | Mazer et al., NEJM 2017. https://www.tctmd.com/news/trics-iii-restrictive-red-cell-transfusion-noninferior-liberal-use-cardiac-surgery (excerpt) |
| CABG variation | n/a | STS 2008, 102,470 CABG: hospital RBC rate **7.8-92.8%**, FFP 0-97.5%, platelets <1->90% | Bennett-Guerrero et al., JAMA 2010;304(14):1568-75. https://scholars.duke.edu/publication/742147 (excerpt) |
| Orthopaedic (THA/TKA) | generator 3.7% | NSQIP: THA 22.2%, TKA 18.3% (older data); Swedish prospective cohort with TXA 16%; single-centre THA fell from 28.5% (2002-03) to 4.3% (2013-20) | https://vivo.weill.cornell.edu/display/pubid25471908 ; https://pmc.ncbi.nlm.nih.gov/articles/PMC4383080 ; https://actaorthopaedica.be/archive/volume-92/summer/original-clinical-research-hip-arthroplasty/perioperative-transfusion-in-total-hip-arthroplasty-predictive-factors-and-outcomes/ (all excerpt) |
| General / elective (India) | generator 3.6% | Indian tertiary centre MSBOS: about **15%** red-cell use across elective surgery, C/T ratio 4.57; NMJI audit of 1,151 elective general surgery patients (procedure-level TP, e.g. MRM 1.5%) | https://impressions.manipal.edu/open-access-archive/4217 ; https://nmji.in/?p=5786 (excerpt) |
| Emergency/trauma | generator 15.0% | US level I centre, 25,599 trauma pts: **about 17%** transfused; multiple-trauma Korea 60.3% | https://experts.arizona.edu/en/publications/blood-transfusions-in-trauma-six-year-analysis-of-the-transfusion/ (excerpt) |
| Dengue platelets (India) | n/a | AIIMS 2013: 531 pts got 1,750 RDP + 114 SDP (about 3.5 units/pt), 23.2% inappropriate; Jaipur 42.6% of dengue pts transfused | https://pmc.ncbi.nlm.nih.gov/articles/PMC4569205 (excerpt) |
| Hb threshold: general | `schema/clinical_thresholds.py:41` 8.0 | **7 g/dL** for stable hospitalised adults (strong rec.) | Carson et al., AABB 2023, JAMA 2023;330(19):1892-1902, doi:10.1001/jama.2023.12914. https://www.isbtweb.org/resource/red-blood-cell-transfusion-2023-aabb-international-guidelines.html (excerpt) |
| Hb threshold: cardiac | `:42` 9.0 | **7.5** cardiac surgery; 8 orthopaedic or pre-existing CVD | same |
| Hb threshold: emergency | `:43` 10.0 | No guideline gives 10 for stable patients; haemorrhage is managed clinically, not by an Hb cut-off | Carson 2023 (above) |
| Platelets: major surgery | `:44` 50,000 | **50 x10^3/uL** for major non-neuraxial surgery; 10k prophylactic (hypoproliferative); 20k LP; 10k CVC at compressible sites; no transfusion for non-bleeding dengue | Metcalf et al., AABB/ICTMG 2025, JAMA, doi:10.1001/jama.2025.7529. https://www.aabb.org/news-resources/news/article/2025/05/29/aabb-develops-new-platelet-transfusion-guidelines (excerpt); older: Kaufman et al., Ann Intern Med 2015;162:205-13 |
| Platelets: cardiac | `:45` 100,000 | 100k is used for neurosurgery/eye in some local protocols; **I found no guideline giving 100k for cardiac surgery** | none found |
| FFP: INR | `:46` INR>1.5 | Used as the appropriateness benchmark in Indian FFP audits (BCSH-based): Jaipur, 62.3% of FFP went to INR>1.5, 39.6% of requests inappropriate. WHO handbook (already in repo) says INR<2.0 before most surgery | https://www.msjonline.org/index.php/ijrms/article/view/4130 (excerpt). **NBTC India document with these thresholds: not found** |

**Recommendation:** the label logic is fine, but surgery type only moves the threshold, never the patient's Hb, so cardiac PRBC (7.8%) comes out 7-9x below TRICS III. **Change:** draw a surgery-specific post-operative Hb drop (e.g. Cardiac about 3-4 g/dL, Ortho about 2-3, General about 1, Emergency/trauma about 2-4 with a wide spread) and apply the threshold to the nadir. Target RBC rates of about 40-60% cardiac, 10-20% ortho, 5-15% general, 15-25% emergency. Re-label thresholds as AABB 2023 (7 / 7.5 cardiac / 8 ortho) or keep the current numbers but call them "local protocol" (the code already does that).

## 4. Demand seasonality (India)

| Parameter | Current (file:line) | Real-world value | Source |
|---|---|---|---|
| Dengue peak timing | `data/demand_process.py:32` Gaussian peak day 258 (~15 Sep), SD 25 days | Peak **Sep-Oct** (Chennai 2006-08; Karwar, Karnataka: rise from June, peak Sep-Oct); Jaipur peak Oct; Bhopal post-monsoon, about 2-month lag after rain. Exception: Mangaluru peaks June-July | Chennai IJMR 2011 https://pmc.ncbi.nlm.nih.gov/articles/PMC3103159 ; Karwar https://microbiologyjournal.org/magnitude-seasonal-variation-serological-and-hematological-profile-of-dengue-in-a-tertiary-teaching-hospital-karwar-india/ ; Mangaluru https://f1000research.com/articles/12-817 (all excerpt) |
| Dengue platelet amplitude | `:35` +90% at peak | **No peer-reviewed monthly magnitude found.** News only: a Jaipur private blood bank went from 15-20 units/month to 50-60/day | https://healthstory.livlong.com/news/increase-in-dengue-cases-leads-to-rising-demand-for-platelets (news, anecdotal) |
| Weekday vs weekend | `:29` 1.15 vs 0.65 (weekend = 0.57x weekday) | UCLH London platelets 2015-16: about 26-33 units weekday vs about 19 weekend (0.58-0.73x). Ontario: strong day-of-week effect in RBC orders (Mon/Fri high) | Farrington et al. 2024 https://arxiv.org/pdf/2411.14939 (figure from search excerpt of PDF; not in abstract); https://arxiv.org/pdf/2008.07486 |
| PRBC yearly wave | `:34` +/-8% sine | **No source found** | none |
| Monsoon/festival trauma | not modelled | **No quantified source found** | none |
| Supply dip: summer | not modelled (no supply seasonality anywhere) | Mumbai (Nanavati, Apr-May 2016): **30-40%** fewer donors (194 vs about 300/month), from college holidays and travel. Karnataka: recurring Mar-May shortage after exams | https://medicaldialogues.in/nanavati-super-speciality-hospital-hit-by-blood-shortage (news, verified) |
| Supply dip: exams/festivals/monsoon | not modelled | Qualitative: "exams and vacations empty university campuses; festivals and fasting reduce participation; summer heat or monsoon illnesses suppress turnout" | ORF, 4 Oct 2025. https://www.orfonline.org/expert-speak/securing-india-s-lifeblood-for-a-reliable-national-blood-supply (verified) |
| National scale | n/a | 2024-25: 14,601,147 units collected (+15% on 12,695,363); demand estimate 14.6M; about 70% voluntary | ORF (above, citing MoHFW) |

**Recommendation:** keep dengue timing (Sep-Oct is right for Karnataka) and the weekday factor. Call the +90% amplitude an assumption. **Add** a supply-side summer dip (Apr-May, about -30%) to donor availability, cited as a news report, plus Ramadan-style dips if you validate on the Malaysia data (section 7).

## 5. Shelf life and storage

| Parameter | Current (file:line) | Real-world value | Source |
|---|---|---|---|
| PRBC | `bloodbank/catalog.py:27` 42 days | RBC in additive solution (SAGM): **42 days** at 1-6 C; CPDA-1 whole blood/RBC: **35 days** | US 21 CFR 610.53 (verified): https://www.law.cornell.edu/cfr/text/21/610.53 |
| Platelets | `:27` 5 days | **5 days at 20-24 C** (with agitation); India: 5 days with anticoagulant, 20-24 C | 21 CFR 610.53 (verified); CAHO India https://caho.in/files/Blood%20and%20Blood%20components.pdf (excerpt) |
| FFP | `:27` 365 days | **1 year** at <= -18 C (US); India: 1 year at -30 C (5 years at -60 C) | 21 CFR 610.53 (verified); CAHO (excerpt) |
| India legal text | n/a | Drugs & Cosmetics Rules 1945, Schedule F Part XII-B; CDSCO guidelines PDF exists but I could not extract its text | https://cdsco.gov.in/opencms/resources/UploadCDSCOWeb/2018/UploadBloodBank/Guidelines_for_blood_bank-merged-with-GSR-166E-dated-11.03.2020.pdf (not read) |

**Recommendation:** keep. Optionally note that CPDA-1 bags give 35 days. Cite 21 CFR 610.53 alongside the WHO handbook already in the repo.

## 6. Donor response and deferral

| Parameter | Current (file:line) | Real-world value | Source |
|---|---|---|---|
| Response probability | `agents/donor_module.py:98` Beta(2,2), **mean 0.50** | Guangzhou RCT, 11,880 lapsed donors: 1-year re-donation **8.1% phone, 8.5% SMS, 7.4% no contact**. Only 40.0% of phone calls reached the donor (20.8% wrong number, 36.1% no answer); SMS delivered 79.3% | Ou-Yang et al., BMC Public Health 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7147048/ (verified) |
| India phone-contact response | n/a | Reactive-donor notification by phone: about **60%** responded to the call (not a donation) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3934289 (excerpt; mapping of figure to URL not fully verified) |
| Monthly donation base rate | n/a | UCI Taiwan donor pool: **23.8%** of 748 donors donated in March 2007 (computed from the file) | UCI dataset (section 7) |
| Active-donor emergency recall (SMS/phone) | n/a | **No solid published rate found** | none |
| Whole-blood interval | `:67` 90 days, all donors | **90 days men, 120 days women** | NBTC donor selection guidelines 2017, via slides: https://www.slideshare.net/slideshow/blood-donor-selection-guidelines-2017/182369567 (official NBTC PDF not retrieved) |
| Apheresis interval | `:68` 28 days, for any donation after apheresis **and apheresis itself** | **48 hours** between platelet/plasma apheresis, at most 2 per week, 24 per year; **28 days** before whole blood after platelet apheresis | same NBTC 2017 slides |
| Response window | `:69` 48 h | Design choice; no source needed | n/a |
| Days since last donation | `:92` uniform 0-300 | Not researched | none |

**Recommendation:** **Change** Beta(2,2) to a lower-mean prior, e.g. Beta(1.5, 6) (mean about 0.2). Label it an assumption bracketed by 8% (lapsed donors, 1 year) and about 24% (UCI monthly) to 60% (answering a call). **Add** donor sex and use 120 days for women. Set apheresis-to-apheresis to 2 days and keep 28 days for apheresis-to-whole-blood.

## 7. Validation datasets

| Dataset | URL | Licence | Size / columns | Use |
|---|---|---|---|---|
| UCI Blood Transfusion Service Center (Hsin-Chu, Taiwan) | page: https://archive.ics.uci.edu/dataset/176/blood+transfusion+service+center ; file (verified 200): https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data | CC BY 4.0, DOI 10.24432/C5GS39 | 748 rows. `Recency (months)`, `Frequency (times)`, `Monetary (c.c. blood)`, `Time (months)`, `whether he/she donated blood in March 2007` (23.8% = 1) | Calibrate and test the donor-response model (RFM features to P(donate)). Cite Yeh, Yang & Ting, "Knowledge discovery on RFM model using Bernoulli sequence", Expert Syst Appl 2009 |
| NHANES 2021-23 CBC + DEMO | CBC_L.xpt / DEMO_L.xpt under https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/ (verified); 2017-18: CBC_J / DEMO_J under /Public/2017/DataFiles/ | US public domain | 8,727 CBC records (7,593 with Hb and platelets); LBXHGB, LBXPLTSI, SEQN, RIAGENDR, RIDAGEYR | Validate Hb/platelet marginals by sex and age (no INR). Use survey weights for population claims |
| MIMIC-IV Clinical Database Demo v2.2 | https://physionet.org/content/mimic-iv-demo/2.2/ | **Open access, ODbL v1.0**, no credentialing (verified) | 100 patients, 15.4 MB, 26 tables. `hosp/labevents` (Hb, platelets, INR), `icu/inputevents` blood products: PRBC itemid 225168 (132 rows, 29 pts), Platelets 225170 (53 rows, 14 pts), FFP 220970 (36 rows, 11 pts) (counted by me) | Real labs-to-transfusion pairs to sanity-check the label model. Tiny, ICU-only, US. Full MIMIC-IV needs credentialing plus CITI training. DOI 10.13026/dp1f-ex47 |
| Malaysia MoH daily blood donations | https://storage.data.gov.my/healthcare/blood_donations.csv (catalogue: https://data.gov.my/data-catalogue/blood_donations) ; richer by state: https://raw.githubusercontent.com/MoH-Malaysia/data-darah-public/main/donations_state.csv | CC BY 4.0 (data.gov.my page; the GitHub repo states no licence) | Daily 2006-01-01 to 2026-10-28, about 106k rows (state file): date, state, daily, blood_a/b/o/ab, location_centre/mobile, type_wholeblood/apheresis_platelet/plasma, donors new/regular | Only open, long, daily blood **supply** series found. Good for testing Prophet and supply seasonality (holiday/Ramadan dips) |
| India e-RaktKosh aggregates on data.gov.in | collected: https://www.data.gov.in/resource/stateuts-wise-number-blood-units-collected-source-e-raktkosh-portal-2018-2022 ; issued: https://www.data.gov.in/resource/stateuts-wise-number-blood-bags-issued-source-e-raktkosh-portal-2018-2022 | Probably GODL-India; **not verified** (page returned 403) | State x year, 2018-2022, from a Rajya Sabha answer; no API | National/state scale check only |
| e-RaktKosh portal (live stock) | https://eraktkosh.mohfw.gov.in | **No public API or bulk-reuse terms found** | Web/app search of stock by state/district/blood bank | Screenshots or manual spot checks only |
| Kaggle / open hospital issue or inventory series | none verified | n/a | **No credible open hospital-level issue/inventory time series found** | n/a |

## 8. Wastage / expiry

| Parameter | Current | Real-world value | Source |
|---|---|---|---|
| Overall wastage baseline | simulation baseline about 1.5%, mostly platelets (per brief; `evaluation/simulate.py:222` counts expiry only) | India tertiary, 2023 data: **3.21% overall** discard; **platelets (RDP) 7.88%**, **PRBC 1.73%** (51% of PRBC discards were DAT/ICT-positive, not expiry), FFP 0.43%, cryo 2.85% | Gaude et al., Bioinformation 2026;22(7):4047-50. https://www.bioinformation.net/022/973206300224047.htm (excerpt) |
| High-wastage Indian centre | same | Chennai tertiary, 2019-23: **18%** of 52,553 components discarded; **97% due to expiry**; platelets were 91.6% of all discards | Haran et al., Cureus 2024;16(5):e59493. https://pmc.ncbi.nlm.nih.gov/articles/PMC11143397 (verified) |
| Other Indian platelet discard | same | W. Uttar Pradesh: platelets 38.4%, FFP 13.7%; ranges in other studies: PRBC 1.7-7.5%, FFP 5-14% | https://www.msjonline.org/index.php/ijrms/article/download/13332/8600/61253 (excerpt; mapping of figures to URL not fully verified) |
| US RBC outdate | same | NBCUS 2021: 305,000 RBC units outdated of 11.78M collected, **about 2.6%** (my division) | Free et al., Transfusion 2023. https://stacks.cdc.gov/view/cdc/133834 (excerpt) |
| Platelet outdate vs shelf life | same | Canada model: each extra day of shelf life cuts wastage about 16%; 5 to 7 days cuts it about 38% | https://pmc.ncbi.nlm.nih.gov/articles/PMC8621808 (excerpt) |

**Recommendation:** 1.5% is at or below the best Indian figure (3.2% overall all-cause; expiry-only would be lower). Platelet-dominated wastage is correct. **Change:** calibrate the "no HemoSmart" baseline so platelet expiry is about 8-15% of platelet units and PRBC about 1-2%. Report wastage per component as well as overall, because the overall figure hides the platelet problem. State that the real figures include non-expiry discards.

---

## Top 5 changes to make the generator more realistic

1. **Weight donor blood types by `GROUP_SHARE`** (`agents/donor_module.py:91`). Uniform draws give 12.5% O- donors vs about 2% real (Agrawal 2014).
2. **Model surgical blood loss.** Add a surgery-specific post-op Hb drop so cardiac RBC rates land near 40-60% (TRICS III 52-73%), ortho 10-20%, general 5-15%, trauma 15-25%. Today cardiac is 7.8%.
3. **Fix INR** (`data/create_dataset.py:43`). Use a right-skewed draw with about 3-5% above 1.5, not 13%. That stops FFP from being the most-ordered component (15.4% now).
4. **Add sex.** Hb differs by about 2 g/dL between men and women (NHANES; WHO 13/12 cut-offs; NFHS-5 57% vs 25% anaemia). Also gives 90/120-day deferral (NBTC 2017).
5. **Lower donor responsiveness and add supply seasonality.** Beta(2,2) (mean 0.5) becomes about mean 0.2, bracketed by 8% (Ou-Yang 2020 RCT) and 24% (UCI). Add an Apr-May supply dip of about 30% (Mumbai report). Fix the apheresis interval to 48 h (NBTC).

Also worth doing: align Hb thresholds to AABB 2023 (7 / 7.5 cardiac / 8 ortho), or keep calling them local protocol. Drop or justify the 100k cardiac platelet threshold, which no guideline I found supports.

## Validation datasets ranked by usefulness

1. **NHANES CBC_L + DEMO_L.** Direct check of Hb/platelet marginals by sex and age; public domain; large.
2. **MIMIC-IV Demo v2.2.** The only open source linking real Hb/platelet/INR to real PRBC/platelet/FFP transfusions (29/14/11 patients). Small and US ICU, so use it as a sanity check, not for training.
3. **UCI Blood Transfusion Service Center.** Real donor return behaviour (23.8% monthly) to calibrate and test the bandit's response model.
4. **Malaysia MoH daily donations (CC BY 4.0).** 20 years of daily supply by blood group; backtest Prophet and supply seasonality on real data.
5. **data.gov.in e-RaktKosh state aggregates.** Coarse annual scale check for India only.

## Not sourced (say so if asked)

- Mean pre-op Hb in Indian surgical patients (no national study found).
- Share of pre-op patients with INR>1.5.
- Hospital share of thrombocytopenic patients (the 8% subgroup).
- Peer-reviewed magnitude of the dengue platelet surge; monsoon/festival trauma effect; PRBC yearly wave.
- The PRBC : platelet : FFP issue ratio behind `BASE_DAILY_UNITS` (20 : 7 : 6).
- Response rate of active donors to SMS/phone emergency appeals.
- An official NBTC/DGHS adult clinical-use guideline with platelet/FFP thresholds (only the NBTC 2017 donor-selection rules, via slides).
- Text of the CDSCO shelf-life rules (PDF could not be parsed here).
