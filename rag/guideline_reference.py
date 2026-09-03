"""
Transfusion Guideline Reference Text
----------------------------------------
This is the knowledge base the RAG engine retrieves from. It is written
from general, widely-known clinical transfusion thresholds (the same
thresholds referenced throughout this project's dataset simulation and
labeling logic), not copied from any single copyrighted document.

IMPORTANT FOR PERSON B: For a real submission/report, replace or
supplement this with the actual WHO Clinical Transfusion Guidelines
PDF (publicly available at who.int) run through the PDF extraction
pipeline -- this placeholder exists so the RAG pipeline can be built
and tested end-to-end right now without waiting on that.
"""

TRANSFUSION_GUIDELINES = """
Red Blood Cell (RBC) Transfusion Guidelines:
Transfusion of red blood cells is generally indicated when hemoglobin
falls below 8 g/dL in surgical patients, or below 7 g/dL in stable,
asymptomatic patients without active bleeding or cardiac disease.
In patients with symptomatic anemia, cardiovascular disease, or active
bleeding, a higher threshold of 9 to 10 g/dL may be appropriate.

Platelet Transfusion Guidelines:
Platelet transfusion is generally indicated when platelet count falls
below 50,000 per microliter prior to major surgery or invasive
procedures. For patients undergoing neurosurgery or procedures with
high bleeding risk, a higher threshold of 100,000 per microliter is
recommended. In stable, non-bleeding patients, prophylactic platelet
transfusion is typically considered below 10,000 per microliter.

Fresh Frozen Plasma (FFP) Guidelines:
FFP transfusion is indicated when INR exceeds 1.5 in patients who are
actively bleeding or require an invasive procedure. FFP is not
recommended solely to correct a mildly elevated INR in the absence of
bleeding or a planned procedure, as this does not reliably reduce
bleeding risk and carries unnecessary transfusion risk.

Emergency and Trauma Transfusion Guidelines:
In trauma and emergency settings, transfusion thresholds are generally
more liberal due to the risk of rapid, ongoing blood loss. A hemoglobin
threshold of 10 g/dL is commonly used in emergency surgical patients,
compared to 8 g/dL in stable elective surgical patients. Massive
transfusion protocols may apply in cases of severe hemorrhage,
involving balanced ratios of red cells, plasma, and platelets.

Pediatric Transfusion Guidelines:
In stable pediatric patients, a more conservative hemoglobin threshold
of 7 g/dL is generally used. In critically ill children or those with
cyanotic heart disease, higher thresholds closer to 9-10 g/dL may be
appropriate.

Chronic Anemia Guidelines:
In patients with chronic, well-compensated anemia (such as those with
thalassemia or chronic kidney disease), transfusion is often deferred
until hemoglobin falls below 6-7 g/dL, or until the patient becomes
symptomatic, since these patients frequently adapt physiologically to
lower hemoglobin levels over time.

Surgical Risk Stratification:
Cardiac and major orthopedic surgeries carry a higher intrinsic risk of
significant blood loss compared to general surgical procedures, and
pre-operative hemoglobin and platelet levels should be assessed closely
in these categories. Emergency surgery carries the least opportunity for
pre-operative optimization and therefore warrants closer monitoring of
all three parameters: hemoglobin, platelet count, and INR.

Risks of Unnecessary Transfusion:
Blood transfusion, while life-saving when indicated, carries risks
including transfusion reactions, transmission of infection, fluid
overload, and alloimmunization. Transfusion decisions should therefore
balance the risk of untreated anemia or coagulopathy against these
transfusion-associated risks, rather than transfusing based on
laboratory values alone without clinical context.
"""