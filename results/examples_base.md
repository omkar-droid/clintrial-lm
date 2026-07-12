# Qualitative examples — base

### Task: condition_qa

**Prompt:** Based on this trial summary, what medical conditions is it studying? List them.

The aim of our study is to determine pain, sensation and electrodiagnostic findings in CTS; to examine its effects on activity and participation.

**Reference:** This trial studies: Carpal Tunnel Syndrome.

**Prediction:** The medical condition studied in this trial is Carpal Tunnel Syndrome (CTS).

---

### Task: phase_classification

**Prompt:** What clinical trial phase is described below? Answer with the phase(s) only.

This research is to understand more about the forces that are involved on the tongue and other oropharyngeal structures during laryngeal surgery. This information will be used to determine if these forces can affect outcomes (pain, numbness, taste, etc) and whether factors such as procedure type, surgeon experience (i.e. resident, fellow, attending), and choice of surgical equipment affect these forces. This informatio

**Reference:** Na

**Prediction:** Phase I/II

---

### Task: plain_language_summary

**Prompt:** Write a short, plain-language summary (3-5 sentences) of this clinical trial for a patient audience.

Title: Phase II Prospective Monocentric Study on Prostate Cancer Restaging by Using PET/MR With Innovative Radiotracers
Conditions: Prostate Cancer
Interventions: 68Ga-PSMA, 68Ga-RM2

**Reference:** The overall goal is to provide an innovative approach to restage patients with biochemical recurrence of prostate cancer by using hybrid PET/MR with innovative radiotracers (68Ga-PSMA and 68Ga-RM2)

**Prediction:** This study is examining new imaging techniques to better assess prostate cancer recurrence. Patients with prostate cancer will receive injections of two special tracers—68Ga-PSMA and 68Ga-RM2—and then undergo PET/MR scans to help doctors more accurately determine if the cancer has returned.

---

### Task: condition_qa

**Prompt:** Based on this trial summary, what medical conditions is it studying? List them.

Early rehabilitation protocols have been studied in Achilles tendon (AT) rupture patients, but deficits in tendon biomechanical properties have been observed several years after the injury. AT rupture patients are unable to return to their previous levels of physical activity. They present deleterious adaptations in the plantar flexor muscles that lead to functional deficits, and deficits in the tendon's structural 

**Reference:** This trial studies: Achilles Tendon Rupture.

**Prediction:** The medical condition being studied is Achilles tendon (AT) rupture.

---

### Task: plain_language_summary

**Prompt:** Write a short, plain-language summary (3-5 sentences) of this clinical trial for a patient audience.

Title: Reversible Secondary Myelofibrosis or Clonal Myeloproliferative Disorder
Conditions: Primary Myelofibrosis, Primary Pulmonary Hypertension, Secondary Myelofibrosis, Pulmonary Arterial Hypertension

**Reference:** To determine the prevalence of myelofibrosis in patients with primary pulmonary hypertension, and to discover if the fibrosis in these patients is primary (AMM) or secondary.

**Prediction:** This clinical trial is studying treatments for people with myelofibrosis or related blood disorders. It aims to evaluate the safety and effectiveness of a new medication in managing symptoms and improving quality of life for participants with primary or secondary myelofibrosis, as well as those with associated pulmonary hypertension.

---

### Task: phase_classification

**Prompt:** What clinical trial phase is described below? Answer with the phase(s) only.

GENERAL OBJECTIVE The general objective is to assess the safety and efficacy of photodynamic therapy (PDT) in the treatment of neurofibromatosis 1 (NF1) tumors in the skin. SPECIFIC OBJECTIVE This is a light dose escalation pilot study to determine the safety and efficacy of PDT using 5-aminolevulinic acid (ALA) and 633 nm light in the treatment of benign dermal neurofibromas. Specifically, the primary goal of the curr

**Reference:** Phase1

**Prediction:** Phase I

---

### Task: extract_eligibility

**Prompt:** Extract the eligibility criteria from the trial text below into JSON with two lists, "inclusion" and "exclusion". Copy each criterion verbatim; do not add any.

Trial eligibility text:
Inclusion Criteria: * Age ranging from 55 to 75 Planned total knee arthroplasty Exclusion Criteria: * Age under 55 or above 75 Patients institutionalized Patients who have received a total knee arthroplasty of the ipsilateral knee Patient not affiliated to a social security scheme (beneficiary or assignee) Patient

**Reference:** {
  "inclusion": [
    "Age ranging from 55 to 75 Planned total knee arthroplasty"
  ],
  "exclusion": [
    "Age under 55 or above 75 Patients institutionalized Patients who have received a total knee arthroplasty of the ipsilateral knee Patient not affiliated to a social security scheme (beneficiary or assignee) Patients with chronic inflammatory rheumatism Cognitive and behavioral issues Disorders of understanding and expression of the French language TKR on complex knee"
  ]
}

**Prediction:** ```json
{
  "inclusion": [
    "Age ranging from 55 to 75",
    "Planned total knee arthroplasty"
  ],
  "exclusion": [
    "Age under 55 or above 75",
    "Patients institutionalized",
    "Patients who have received a total knee arthroplasty of the ipsilateral knee",
    "Patient not affiliated to a social security scheme (beneficiary or assignee)",
    "Patients with chronic inflammatory rheumatism",
    "Cognitive and behavioral issues",
    "Disorders of understanding and expression of the 

---

### Task: extract_eligibility

**Prompt:** Extract the eligibility criteria from the trial text below into JSON with two lists, "inclusion" and "exclusion". Copy each criterion verbatim; do not add any.

Trial eligibility text:
Inclusion Criteria: 1. Willing to attend 3 follow-up visits during the intervention period 2. Agree to provide blood, urine, and stool samples twice during the intervention period 3. Good eyesight, able to read and write, and capable of wearing glasses if needed 4. Good hearing and able to fully comprehend all ins

**Reference:** {
  "inclusion": [
    "Willing to attend 3 follow-up visits during the intervention period",
    "Agree to provide blood, urine, and stool samples twice during the intervention period",
    "Good eyesight, able to read and write, and capable of wearing glasses if needed",
    "Good hearing and able to fully comprehend all instructions during the intervention"
  ],
  "exclusion": [
    "Presence of digestive diseases, particularly gastrointestinal disorders (e.g., celiac disease, ulcerative coli

**Prediction:** ```json
{
  "inclusion": [
    "Willing to attend 3 follow-up visits during the intervention period",
    "Agree to provide blood, urine, and stool samples twice during the intervention period",
    "Good eyesight, able to read and write, and capable of wearing glasses if needed",
    "Good hearing and able to fully comprehend all instructions during the intervention"
  ],
  "exclusion": [
    "Presence of digestive diseases, particularly gastrointestinal disorders (e.g., celiac disease, ulcerat

---

### Task: phase_classification

**Prompt:** What clinical trial phase is described below? Answer with the phase(s) only.

The purpose of this trial is to determine whether treatment with valsartan will have beneficial effect in early hypertrophic cardiomyopathy (HCM) by assessing many domains that reflect myocardial structure, function and biochemistry.

**Reference:** Phase2

**Prediction:** Phase II

---

### Task: extract_eligibility

**Prompt:** Extract the eligibility criteria from the trial text below into JSON with two lists, "inclusion" and "exclusion". Copy each criterion verbatim; do not add any.

Trial eligibility text:
Inclusion criteria: * Age: 18-65 years old. * Gender: Male and female. * Newly diagnosed colorectal cancer patients who are scheduled to receive capecitabine-based chemotherapy. * Eastern Cooperative Oncology Group (ECOG) performance status \< 2 * Patients with adequate renal function (Sr. creatinine \< 1.2 mg/dl 

**Reference:** {
  "inclusion": [
    "Age: 18-65 years old.",
    "Gender: Male and female.",
    "Newly diagnosed colorectal cancer patients who are scheduled to receive capecitabine-based chemotherapy.",
    "Eastern Cooperative Oncology Group (ECOG) performance status \\< 2",
    "Patients with adequate renal function (Sr. creatinine \\< 1.2 mg/dl or eGFR \u2265 90 ml/min).",
    "Patients with adequate hepatic function (Sr. bilirubin \\< 1.2 mg/dl)."
  ],
  "exclusion": [
    "Pregnant and lactating femal

**Prediction:** ```json
{
  "inclusion": [
    "Age: 18-65 years old.",
    "Gender: Male and female.",
    "Newly diagnosed colorectal cancer patients who are scheduled to receive capecitabine-based chemotherapy.",
    "Eastern Cooperative Oncology Group (ECOG) performance status < 2",
    "Patients with adequate renal function (Sr. creatinine < 1.2 mg/dl or eGFR ≥ 90 ml/min).",
    "Patients with adequate hepatic function (Sr. bilirubin < 1.2 mg/dl)"
  ],
  "exclusion": [
    "Pregnant and lactating females."

---

### Task: plain_language_summary

**Prompt:** Write a short, plain-language summary (3-5 sentences) of this clinical trial for a patient audience.

Title: Targeting Non-Luminal Disease by PAM50 With Pembrolizumab + Paclitaxel in Hormone Receptor-positive/HER2-negative Advanced/Metastatic Breast Cancer, Who Have Progressed on or After CDK 4/6 Inhibitor Treatment
Conditions: Metastatic Breast Cancer
Interventions: Pembrolizumab, Paclitaxel

**Reference:** This is an open-label, single arm, multicenter phase II study evaluating treatment with pembrolizumab in combination with paclitaxel in patients with locally advanced or metastatic non-luminal hormone receptor-positive (HR+), human epidermal growth factor receptor 2-negative (HER2-) (hereafter referred to as HR+/HER2-) breast cancer who had recurrence or progression while receiving previous therapy with a cyclin-dependent kinase (CDK) inhibitor in the adjuvant setting or to treat advanced diseas

**Prediction:** This clinical trial is testing a new treatment approach for women with advanced or metastatic breast cancer that no longer responds to CDK 4/6 inhibitors. The study combines the drugs pembrolizumab and paclitaxel, targeting a specific subtype of breast cancer that does not rely on luminal characteristics. Participants must have hormone receptor-positive and HER2-negative breast cancer.

---

### Task: extract_eligibility

**Prompt:** Extract the eligibility criteria from the trial text below into JSON with two lists, "inclusion" and "exclusion". Copy each criterion verbatim; do not add any.

Trial eligibility text:
Inclusion Criteria: * patients aged 35 to 85, * ASA 2-3 (according to the classification of American Society of Anesthesiologists), * undergoing a planned laparoscopic intestine resection at the Department of Abdominal Surgery at the University Medical Centre Ljubljana. Exclusion Criteria: * allergies to alpha 2 r

**Reference:** {
  "inclusion": [
    "patients aged 35 to 85,",
    "ASA 2-3 (according to the classification of American Society of Anesthesiologists),",
    "undergoing a planned laparoscopic intestine resection at the Department of Abdominal Surgery at the University Medical Centre Ljubljana."
  ],
  "exclusion": [
    "allergies to alpha 2 receptor agonists,",
    "uncontrolled arterial hypertension,",
    "2nd and 3rd degree atrioventricular block,",
    "alcohol and illegal drugs abusers,",
    "patient

**Prediction:** ```json
{
  "inclusion": [
    "patients aged 35 to 85",
    "ASA 2-3 (according to the classification of American Society of Anesthesiologists)",
    "undergoing a planned laparoscopic intestine resection at the Department of Abdominal Surgery at the University Medical Centre Ljubljana"
  ],
  "exclusion": [
    "allergies to alpha 2 receptor agonists",
    "uncontrolled arterial hypertension",
    "2nd and 3rd degree atrioventricular block",
    "alcohol and illegal drugs abusers",
    "patien

---

