Condensed documentation: Parsing + normalization + Elo pipeline (SONA + ARIADNA)

Goal

Combine two Qualtrics datasets with slightly inconsistent dimension labels and compute Elo rankings per latent (positive-pole) dimension from paired comparisons stored in ratings_list and ratings_list2.

⸻

Data format assumptions

1) Pairwise comparison encoding

Each ratings_list / ratings_list2 cell contains nested lists of comparisons. Each comparison has the form:

[[label, image_A, a], [label, image_B, b]]

where a and b are binary indicators and exactly one is 1 and the other 0.

Interpretation: the image with value 1 was chosen as “more label”.

2) Positive/negative dimension metadata

Each dataset has two columns:
	•	dims: list of positive-pole labels (e.g., "inteligentna (smart)")
	•	dims_negated: list of corresponding negative-pole labels (e.g., "nieinteligentna (not smart)")

These are used to build:
	•	pos_to_neg: positive → negative label
	•	neg_to_pos: negative → positive label

⸻

Core problem addressed

Across datasets (and sometimes within a dataset), the same concept may appear under different strings inside ratings_list / ratings_list2, e.g.:
	•	"nieinteligentna (smart)" vs "nieinteligentna (not smart)"
	•	"nieurocza (cute)" vs "nieurocza (not cute)"
	•	"szczupła" vs "niegruba" etc.

If untreated, these become separate dimensions.

⸻

Canonicalization strategy

All analysis is done on canonical positive-pole dimension names.

Step A — Resolve each raw label to (base_dim, sign)

A function resolve_label_to_dimension_and_sign(raw_label, pos_to_neg, neg_to_pos) returns:
	•	base_dim: canonical positive dimension label (e.g., "inteligentna (smart)")
	•	sign:
	•	+1 if raw_label is a positive phrasing of base_dim
	•	-1 if raw_label is a negative phrasing (opposite pole)

Resolution order:
	1.	If raw_label ∈ pos_to_neg: (raw_label, +1)
	2.	Else if raw_label ∈ neg_to_pos: (neg_to_pos[raw_label], -1)
	3.	Else if raw_label ∈ ALIAS_LABEL_MAP: use predefined alias (base_dim, sign)
	4.	Else fallback: (raw_label, +1) (keeps unexpected labels visible)

Step B — Alias dictionary

ALIAS_LABEL_MAP contains known mismatches and maps them into canonical dims, e.g.:
	•	"nieinteligentna (smart)" → ("inteligentna (smart)", -1)
	•	"nieurocza (cute)" → ("urocza (cute)", -1)
	•	"szczupła" → ("gruba", -1)
	•	"skromna" → ("zarozumiała", -1)
	•	and spelling variants like Middle East negation forms

⸻

Comparisons dataframe construction

Output schema

A long-format comparisons_df is produced with one row per pairwise decision:
	•	dataset_id: "sona" / "ariadna"
	•	respondent_id
	•	dimension: canonical positive dimension (base_dim)
	•	image_A, image_B
	•	winner: "A" or "B" meaning higher on the positive pole
	•	optional debugging: raw_label, raw_col

Winner computation rule

For each comparison, we know which side is “more raw_label” via the 1:
	•	If sign == +1 (raw_label is positive):
winner = argmax(raw_label) (the image with 1)
	•	If sign == -1 (raw_label is negative):
winner = opposite(argmax(raw_label)) because “more NOT-X” means “less X”

This ensures that every row is oriented consistently: "winner" always means more of the positive trait.

⸻

Elo computation

Inputs

For a given canonical dimension, build match list:
	•	If winner == "A": (image_A, image_B) meaning A beats B
	•	If winner == "B": (image_B, image_A) meaning B beats A

Elo update

Standard Elo updates with parameters:
	•	base rating: 1500
	•	K-factor: 32
	•	multiple passes (epochs): default 5
	•	shuffled order per epoch (seeded)

Expected score:
	•	E_A = 1 / (1 + 10^((R_B - R_A)/400))

Update:
	•	R_A ← R_A + K*(S_A - E_A) where S_A=1 if A wins else 0
	•	R_B ← R_B + K*(S_B - E_B)

Output per dimension:
	•	image, elo, comparisons (count of matches participated in)

⸻

Optional simulation test (sanity check)

To validate that the Elo implementation recovers a known ordering:
	1.	Choose a dimension and take the real (image_A, image_B) pairs.
	2.	Create a random “true” ranking over images.
	3.	Set simulated winners deterministically by that ranking (no noise).
	4.	Run Elo on the simulated outcomes.
	5.	Compare Elo ranking vs true ranking using rank correlation.

If correlation is strong (close to ±1 depending on convention), the Elo pipeline is functioning correctly.

⸻

Outputs
	•	comparisons_df: combined, normalized pairwise decisions across datasets.
	•	elo_all: Elo ratings for every canonical dimension, optionally with URL column:
url = IMAGE_URL_PREFIX + image.

This pipeline ensures that label inconsistencies do not split dimensions and that both ratings_list and ratings_list2 contribute correctly to the same latent trait scale.