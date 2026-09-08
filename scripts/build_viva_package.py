from pathlib import Path
out=Path("output/viva_package");out.mkdir(parents=True,exist_ok=True)
def w(n,x):(out/n).write_text(x.strip()+"\n",encoding="utf8")
L="""## Evidence legend
[M] Main manuscript PDF and paper/manuscript.tex. [S] Supplementary PDF and paper/supplementary.tex. [C] Complete CHANGELOG.md. [R] Frozen artifacts in results and research result folders. [I] Cautious interpretation. Do not add details absent from these sources: NOT SUPPORTED BY THE PROVIDED MATERIAL.
"""
SRC="""## Source map
Final claims: [M] Sections I-VII and appendices. Data/split: [M] III-A/Table I. Ladder and tests: [R] results/ablation_table.csv, bootstrap_cis.json, mcnemar_delong.json. Frozen test: [R] results/session9. Development history: [C] S1-S22. External work: [M] IV-E/Table VIII and [R] results/external.
"""
E="""# Executive understanding
%s
## The paper in 30 seconds
The study tests whether a seven-class dermoscopy classifier is credible beyond a benchmark score. It uses a six-CNN HAM10000 ensemble, then audits calibration, abstention, conformal sets, subgroup outcomes and referral cost. Macro-F1 reaches .8047 and ECE falls .1575 to .0206. The critical result is hidden under-40 failure: sensitivity .143, 3/21, versus .764 at 60+, and abstention refers only 2/18 young misses. Lambda raises total sensitivity .731 to .831 but young sensitivity only .143 to .238 and NNB at 3 percent prevalence 3.0 to 6.2. This is a bounded audit and mitigation, not clinical deployment. [M Abstract; IV-C-D]

## The paper in 2 minutes
NV is 66.8 percent of the test set, so accuracy can reward mole guessing while missing escalating lesions. The project uses lesion-grouped splits and Macro-F1, then separately evaluates AKIEC/BCC/MEL escalation sensitivity. Uniform soft voting was selected on validation; ridge stacking is rejected despite better test score because its validation/OOF score is worse. The ensemble is underconfident. Dirichlet improves calibration but lowers screening sensitivity. Margin referral and class-conditional conformal sets help at aggregate level, yet young patients remain unprotected. Frozen external testing transports the intervention direction while rejecting the proposed mechanism. [M III-IV; C S9/S14]

## One-sentence thesis
Aggregate benchmark performance and marginal guarantees can conceal a severe, confidently wrong, only partly mitigable young-patient escalation failure.

## Problem -> method -> evidence -> conclusion
Problem: benchmark score can hide serious misses. Method: lesion-grouped ensemble, TTA, calibration, referral, conformal and subgroup/external audits. Evidence: ensemble gain is the only certified ladder increment; calibration has sensitivity cost; lambda has referral cost and leaves young failure. Conclusion: auditing and limited mitigation are supported; deployment, causal mechanism and universal fairness are not.
"""%L+SRC
ST="""# Research story
%s
| Phase | Change | Result | Decision |
|---|---|---|---|
| Baselines | Six CNNs | ConvNeXt-Tiny .7459 | Keep controlled benchmark |
| Ensembling | Soft vote/ridge/other methods | Uniform .7718; ridge higher test, worse val | Retain uniform; reject test selection |
| TTA/calibration | 24 views/Dirichlet | .7859; ECE .0206 | Retain, state sensitivity trade-off |
| Extra models | Transformer, fusion, losses | No decisive gain | Report negatives |
| Safety layers | Margin/conformal/fairness | Aggregate help; age failure appears | Build OOF path |
| Lambda | OOF age rule | Under-40 .26, interval includes 0 | Pre-register test |
| Test pass | 19 frozen quantities | Young mechanism and orthogonality weaken | Revise claims |
| External | PAD/BCN/MSKCC | Mechanism fails; direction transports | Report contingency |

## Discrepancy ledger
| Earlier/tempting claim | Final record | Say this instead |
|---|---|---|
| Pure decision-rule young failure | Test young AUC .810 is lower | Both decision-rule and ranking failure |
| Under-40 always worst calibrated | Test 40-59 gap is largest | Sign disagreement stable, rank order unstable |
| Lambda and abstention independent | Young Jaccard=1.00 | Not independent in target subgroup |
| PAD validates intended use | Smartphone modality shift | Shift benchmark, not clinical validation |
| Lambda proves prior mechanism | Both external claims fail | Operating point transports; explanation does not |
| OOF bipartite is guarantee | Nonstandard split setting | Empirical audit, not theorem |
"""%L+SRC
M="""# Method deep dive
%s
## Data and task
HAM10000: 10,015 images, 7,470 lesions; lesion-ID train/validation/test split 6,981/1,532/1,502. Image-level split would leak near-duplicate lesions. Seven classes: AKIEC 52, BCC 71, BKL 167, DF 20, MEL 167, NV 1004, VASC 21 on test. Escalating set is AKIEC/BCC/MEL. NV is 66.8 percent, so Macro-F1 is preferable to accuracy. [M III-A/Table I]

## Pipeline
Six CNNs: ConvNeXt-Tiny/Small, DenseNet-121, EfficientNet-B0/B3, ResNet-50. Supported details: ImageNet initialization, three frozen-head epochs then fine-tune, effective-number class weighting and seed 42. Exact optimizer, LR, input resolution and unlisted augmentation are NOT SUPPORTED BY THE PROVIDED MATERIAL.

Image -> six probability vectors -> uniform average -> 24 TTA views -> Dirichlet -> diagnosis, margin referral, conformal set and lambda decision rule. Uniform vote needs no fitted weights. Ridge .7815 test is rejected because uniform .7911 validation/OOF is better versus ridge .7583. [R ensembling]

## Calibration
Confidence .7048 versus accuracy .8609 shows underconfidence, ECE .1575. Dirichlet test ECE .0206; temperature .0322; matrix .0279. Dirichlet improves TTA Macro-F1 .7859 to .8047 but sensitivity falls .7862 to .7310 and missed serious rises 62 to 78. Calibration and screening safety are different objectives. [M IV-A]

## Abstention/conformal
Margin is top probability minus runner-up. At 10 percent target abstention, 89.2 percent retained with Macro-F1 .8577 and sensitivity .7621 among retained cases. Mahalanobis loses ID ranking but detects PAD shift, AUROC about .913. LAC marginal alpha .10 has .904 coverage overall but .748 serious coverage and 63 false reassurances. Class-conditional RAPS has .941 serious coverage and 6 FRR events, mean set size 2.68. OOF bipartite coverage is empirical, not a split-conformal theorem. [M IV-B; R selective]

## Lambda and NNB
y_hat(x)=argmax_c[p_c(x)+lambda_band(x)*1(c escalating)]. This is post-processing, not retraining. Lambda values: under-40 .26, 40-59 .74, 60+ .33, unknown pooled .65. Young CI [.00,.61]. OOF has 64 young positives versus validation 22, enabling the 30-positive fit gate. One scalar per band controls overfitting. NNB(pi)=(TP+wFP)/TP; at pi=.03 lambda changes 3.0 to 6.2. [M III-C/App. A]
"""%L+SRC
R="""# Results deep dive
%s
## Ablation ladder
| Rung | Change | Macro-F1 | Sensitivity | Decision |
|---|---|---:|---:|---|
| A2 | ConvNeXt-Tiny | .7459 | .7759 | Best individual |
| A5 | Uniform six-CNN vote | .7718 | .7828 | Only certified gain |
| A6 | A5 plus TTA | .7859 | .7862 | Point improvement |
| A7-val | A6 plus validation Dirichlet | .8047 | .7310 | Best classification result, safety trade-off |
| A7-oof | OOF calibration | .7871 | .7310 | Negative rung, Holm .242 |
| A8 | 30-member fold bag | .7810 | .7345 | Negative rung, Holm .242 |

A5 versus A2: McNemar Holm p=2.0e-4. BKL AUC DeLong Holm p=.0129. McNemar and Macro-F1 bootstrap test different estimands. [R mcnemar_delong; C S7]

## What worked
Uniform ensemble, TTA point gain, Dirichlet probability calibration, margin referral for ID difficulty, class-conditional conformal safety, lambda overall sensitivity direction, and Mahalanobis OOD separation.

## What failed
Ridge as reported winner; transformer/fusion/loss variants; greedy selection/prior correction/fold bag; cost threshold as ladder; marginal conformal safety; pure decision-rule age story; prior-skew external mechanism; EM PAD correction.

## Numbers to explain
Under-40 .143 [.030,.363], 3/21; 60+ .764; difference -.621 Holm p<.001. Lambda total .731->.831, missed 78->49, young .143->.238, NNB .03 3.0->6.2. Marginal LAC test .901 total/.735 serious. RAPS bipartite alpha .05 FRR .0138 is OOF empirical audit.

## Figure/table guide
Fig.1 pipeline. Fig.2 reliability bars above diagonal show underconfidence, not sensitivity. Risk-coverage: x answered coverage, y retained error. Conformal plot: class coverage, not automatic age coverage. External dose response: failed pre-registered mechanism, no post-hoc causal trend. Decision curves: fixed argmax/lambda versus thresholded risk model; young benefit is null/negative at high threshold. Grad-CAM lesion focus is not causal reasoning. [M Figs.1-7/Tables I-VIII]
"""%L+SRC
C="""# Claim audit
%s
| Claim | Status | Boundary |
|---|---|---|
| Ensemble improves baseline | SUPPORTED | .7459->.7718; Holm .0002 |
| A7 best Macro-F1 | SUPPORTED | .8047, not best screening policy |
| Dirichlet improves ECE | SUPPORTED | .1575->.0206, sensitivity cost |
| Calibration improves screening safety | NOT SUPPORTED | sensitivity .7862->.7310 |
| Marginal coverage protects serious | NOT SUPPORTED | .901 total/.735 serious |
| Class conditioning helps | SUPPORTED | longer sets |
| OOF bipartite theorem | NOT SUPPORTED | empirical only |
| Under-40 failure | SUPPORTED | .143 versus .764 |
| Pure decision-rule explanation | FAILED | young AUC .810 lower |
| Lambda total sensitivity | SUPPORTED | .731->.831 |
| Lambda solves young gap | NOT SUPPORTED | .143->.238 |
| PAD clinical validation | NOT SUPPORTED | modality shift |
| Prior mechanism transports | FAILED | external claims fail |
| Lambda direction transports | PARTIAL | explanation/threshold do not |
| Skin-tone fairness all groups | NOT SUPPORTED | missing/underpowered |
| Grad-CAM causal proof | EXPLORATORY | no causal claim |

## Do not say
The model is deployable; lambda solved young melanoma triage; PAD is clinical validation; 90-percent marginal coverage protects every group; OOF coverage is formal guarantee; prior skew was proven causal; Grad-CAM proves mechanism; or all skin-tone fairness is established.
"""%L+SRC
X="""# Limitations
%s
## Critical
1. No prospective clinical validation or workflow impact study.
2. Only 21 under-40 escalating test lesions; direction strong, precise magnitude uncertain.
3. PAD mixes optical and prevalence shift; external same-modality mechanism test fails.
4. Lambda is post-processing with more referrals, lower Macro-F1 and insufficient young benefit.

## High
Global calibration has opposite-sign age residuals. OOF/external conformal lacks standard theorem. Labels lack inter-rater variability. Skin-tone evidence is incomplete and PAD V/VI underpowered.

## Moderate/minor
DF/VASC intervals are wide; historical training histories were clobbered; external battery required remediation; plan is not registration; colour constancy absent; no CONSORT flow diagram; some stages lacked local LaTex compilation.

## Professional response
“Yes, that is a major limitation. I bound it rather than dismiss it. Lesion splits, frozen fitting, intervals and external audits reduce avoidable bias but do not remove this limitation. A prospective, age-balanced, multi-centre dermoscopy study with clinical workflow outcomes is needed before deployment.”
"""%L+SRC
P="""# Presentation structure and speaker script
%s
## 15 slides
1 Title/thesis. 2 Clinical problem/NV imbalance. 3 Research gap. 4 Questions/contributions. 5 Data and lesion split. 6 Pipeline. 7 Ablation ladder. 8 Calibration trade-off. 9 Abstention/conformal. 10 Under-40 failure. 11 Lambda rule/NNB. 12 External evaluation. 13 What worked/failed. 14 Limitations. 15 Conclusion.

## Script
“Good morning. This work asks whether a dermoscopy classifier that looks strong on a benchmark remains credible after calibration, referral, subgroup performance and clinical cost are measured. NV dominates the test set, so accuracy can reward mole guessing. We use lesion-level splits, Macro-F1 and escalation sensitivity.

Six CNNs are averaged, then 24-view TTA and Dirichlet calibration are applied. Macro-F1 rises .7459 to .7718 to .7859 to .8047. Ridge had a higher test score but lost on validation, so I rejected it. The ensemble is underconfident; Dirichlet improves ECE .1575 to .0206 but sensitivity falls .7862 to .7310.

Margin referral and class-conditional conformal sets improve aggregate safety behavior, but marginal coverage can be 90 percent overall and only .735 for serious lesions. Under 40 sensitivity is .143, 3/21, versus .764 at 60+. Test AUC shows both decision-rule and ranking limitation. Lambda raises total sensitivity .731 to .831, but young sensitivity only .143 to .238 and NNB .03 3.0 to 6.2.

PAD is smartphone shift, not clinical validation. BCN/MSKCC reject the mechanism while frozen lambda direction transports. The operating point transports; the explanation does not. The contribution is an audit and quantified trade-off, not a clinical product.”
"""%L+SRC
Q="""# Quick revision and one-page cheat sheet
%s
Dataset: HAM10000 10,015 images, 7,470 lesions. Split: 6,981/1,532/1,502 by lesion ID. Escalating: AKIEC/BCC/MEL. Pipeline: six CNN vote -> 24 TTA -> Dirichlet. Best Macro-F1 .8047. ECE .1575->.0206. Young .143, 3/21; 60+ .764. Lambda total .731->.831; young .143->.238; NNB .03 3.0->6.2. External: direction transports, mechanism fails. Final: audit/mitigation, no deployment.

## 10 numbers
10,015; 7,470; 6,981/1,532/1,502; NV 66.8 percent; .7459; .7718; .8047; .1575->.0206; .143 versus .764; .731->.831 and 3.0->6.2.

## 10 concepts
Lesion leakage; Macro-F1; validation versus test selection; underconfidence; calibration versus sensitivity; abstention; marginal versus conditional coverage; escalation-mass AUC; lambda post-processing; operating-point transport versus mechanism.

## If I only have 30 minutes
Learn data/classes/split; explain why accuracy/image split fail; rehearse .7459/.7718/.7859/.8047; explain calibration trade-off; memorize 3/21 and caveat; explain lambda/NNB; state marginal coverage is not subgroup protection; PAD is shift test; close with no deployment claim.
"""%L+SRC
V="""# Visual plan
%s
1 Entire paper: problem -> pipeline -> audit -> young failure -> limited lambda -> external qualification.
2 Dataset: HAM10000 -> lesion grouping -> split -> external cohorts.
3 Pipeline: CNNs -> vote -> TTA -> Dirichlet -> diagnosis/referral/set/lambda.
4 Ladder: .7459/.7718/.7859/.8047 with red sensitivity warning.
5 Reliability: ECE before/after plus sensitivity arrow.
6 Risk-coverage: abstention helps average uncertainty.
7 Conformal: overall versus serious coverage.
8 Age bars: .143/.814/.764 hidden stratification.
9 Decision versus ranking: argmax discards some signal, AUC .810 shows information loss.
10 Lambda: total/young sensitivity plus NNB.
11 External: PAD shift vs BCN/MSKCC failed mechanisms.
12 Conclusion: defendable claims vs prohibited overclaims.

~~~mermaid
flowchart LR
A[Lesion grouped HAM10000]-->B[Six CNNs]-->C[Uniform vote]-->D[24 TTA]-->E[Dirichlet]
E-->F[Diagnosis]
E-->G[Referral]
E-->H[Conformal set]
E-->I[Lambda]
~~~
"""%L+SRC
terms=["CNN","Macro-F1","soft voting","TTA","calibration","ECE","Dirichlet calibration","selective classification","risk-coverage","Mahalanobis distance","OOD","conformal prediction","LAC","APS","RAPS","Mondrian calibration","marginal coverage","false reassurance rate","hidden stratification","escalation mass","argmax","lambda rule","OOF","cross-fitting","bootstrap","Clopper-Pearson","McNemar","DeLong","Holm-Bonferroni","prior shift","optical shift","data leakage"]
G="# Glossary\n"+L+"\n"
for t in terms:G+="## "+t+"\n\n**Definition:** See Method Deep Dive for the paper-specific definition and boundary.\n\n**Why it matters here:** "+t+" is one of the concepts that separates raw classifier performance from clinically relevant audit evidence.\n"
G+=SRC
basic=["Why not accuracy?","Why lesion split?","What are escalating classes?","What is soft vote?","What is TTA?","What is calibration?","What is ECE?","What is abstention?","What is conformal prediction?","What is hidden stratification?","What is lambda?","What is NNB?","What is best Macro-F1?","What is young failure?","What is main limitation?","What is PAD?","What is OOD?","What is Macro-F1?","What is FRR?","What is final conclusion?"]
tech=["Why reject ridge?","Why uniform weights?","Why underconfidence?","Why Dirichlet?","Why sensitivity falls?","Why margin?","Why Mahalanobis OOD?","What is risk-coverage?","What is marginal coverage?","What is Mondrian?","What is APS?","What is RAPS?","Why OOF lambda?","Why one lambda?","Why specificity floor?","Why cross-fit?","Why bootstrap lesions?","Why Clopper-Pearson?","What does McNemar test?","What does DeLong test?","Why Holm?","What is escalation AUC?","What is signed gap?","What is prior shift?","What is optical shift?","Why EM fails?","Why no skin-tone claim?","Why Grad-CAM noncausal?","Why foldbag negative?","Why test only once?"]
hard=["Is .143 reliable?","Did lambda solve young failure?","Is age causal?","Is age rule fair?","Why not retrain?","Are lambda and abstention independent?","Is failure only decision rule?","Does lambda prove mechanism?","Is PAD validation?","Can conformal ensure safety?","Does low ECE prove safety?","Does significance prove clinical utility?","Why bootstrap/McNemar differ?","What invalidates test discipline?","Why mechanism fails?","What transports?","What would reviewer praise?","What would reviewer attack?","What is weakest claim?","What next experiment?"]
trap=["Did test choose model?","Does 90 percent cover every group?","Did calibration improve screening?","Was young always worst calibrated?","Did abstention rescue young errors?","Did lambda prove age cause?","Did you prove fairness?","Did EM solve PAD?","Did Grad-CAM prove mechanism?","Is it deployable?"]
def answer_for(q):
 q=q.lower()
 if "accuracy" in q:return "Accuracy is misleading because NV is 66.8 percent of test images; Macro-F1 and escalation sensitivity are therefore primary endpoints."
 if "lesion" in q and "split" in q:return "The split is by lesion ID because image-level splitting could put near-duplicate views of the same lesion into evaluation."
 if "escalating classes" in q:return "AKIEC, BCC and MEL form the escalating set used in binary screening metrics."
 if "soft vote" in q or "uniform weights" in q:return "The six class-probability vectors are averaged equally; equal weighting was validation-leading and avoided fitted-weight selection risk."
 if "ridge" in q:return "Ridge stacking had better observed test Macro-F1 but worse validation/OOF Macro-F1, so it was rejected to avoid test-set model selection."
 if "tta" in q:return "TTA uses eight dihedral transforms and three scales, giving 24 inference views that are pooled without changing model weights."
 if "underconfidence" in q:return "The ensemble is underconfident because averaging disagreement lowers the maximum score: confidence .7048 versus accuracy .8609."
 if "dirichlet" in q:return "Dirichlet handles class-dependent calibration distortion and was selected by validation ECE; its test ECE is .0206."
 if "sensitivity" in q and ("fall" in q or "calibration" in q):return "Calibration shifts some borderline decisions toward common NV, improving ECE while the TTA system sensitivity falls .7862 to .7310."
 if "margin" in q:return "Margin is the top probability minus the runner-up; low margin is direct decision ambiguity and was selected by validation AURC."
 if "mahalanobis" in q:return "Mahalanobis is poor for in-distribution error ranking but useful for PAD modality shift, where AUROC is about .913."
 if "conformal" in q:return "Conformal prediction returns a label set. Standard split conformal gives marginal coverage under exchangeability; OOF bipartite results here are empirical, not theorem-level."
 if "coverage" in q:return "Marginal coverage averages all cases. The test result .901 overall but .735 on serious lesions shows why it cannot be treated as subgroup protection."
 if "aps" in q:return "APS forms a set from cumulative probability mass; RAPS adds a regularization penalty to control set growth."
 if "raps" in q:return "RAPS is regularized APS; class-conditional RAPS produces strong serious coverage but larger sets."
 if "lambda" in q:return "Lambda adds a fixed age-band bonus to escalating classes before argmax. It is post-processing, not retraining; total sensitivity rises .731 to .831, young .143 to .238."
 if "young" in q or ".143" in q:return "Under-40 test sensitivity is .143, 3/21, versus .764 at 60+. The direction is compelling but the exact rate has wide .030-.363 interval."
 if "boot" in q:return "Lesion-grouped bootstrap resamples lesions rather than correlated images; Clopper-Pearson is used for sparse binomial proportions such as 3/21."
 if "mcnemar" in q:return "McNemar compares paired correctness using discordant predictions; the ensemble comparison survives Holm with p=2.0e-4."
 if "delong" in q:return "DeLong compares paired ROC-AUCs; BKL AUC A5 versus A2 survives Holm at p=.0129."
 if "holm" in q:return "Holm-Bonferroni controls familywise error across a pre-declared comparison family."
 if "prior" in q or "optical" in q or "em" in q:return "PAD combines a large prevalence shift with optical smartphone shift. Deployable EM prior correction worsened Macro-F1, showing prior adjustment is insufficient."
 if "pad" in q:return "PAD is a smartphone-photography shift benchmark, not clinical validation of intended dermoscopic use."
 if "fair" in q or "skin" in q:return "The material does not support broad skin-tone fairness claims: HAM lacks Fitzpatrick labels and PAD V/VI strata are underpowered."
 if "grad" in q:return "Grad-CAM is qualitative support only; prediction-targeted maps do not prove causal lesion reasoning."
 if "external" in q or "transport" in q or "mechanism" in q:return "Frozen lambda raises young sensitivity direction at all three centres, but both pre-registered mechanism claims fail. The operating point transports; the explanation does not."
 if "deploy" in q:return "No. The study is retrospective, subgroup support is sparse, and no prospective clinical workflow validation exists."
 return "State the relevant final estimate, its split and denominator, then name the caveat. Do not turn aggregate or exploratory evidence into a clinical or causal claim."

def section(title,qs):
 x="## "+title+"\n"
 for i,q in enumerate(qs,1):
  a=answer_for(q)
  x+="\n### %d. %s\n\n**Ideal answer:** %s\n\n**Why correct:** It distinguishes the final result from a broader clinical or causal interpretation.\n\n**Common mistake:** Overstatement, omitted denominator/split, or confusing calibration, screening, coverage and clinical utility.\n\n**Follow-up:** What evidence would make the claim stronger?\n\n**10-second version:** %s\n\n**60-second version:** %s State the relevant split and caveat: this is a retrospective frozen-protocol result, not prospective clinical benefit. [M; C S9]\n"%(i,q,a,a.split('.')[0]+'.',a)
 return x
B="# Viva question bank\n"+L+"\nThis bank has 80 questions: 20 basic, 30 technical, 20 difficult and 10 traps.\n"+section("Level 1 Basic",basic)+section("Level 2 Technical",tech)+section("Level 3 Difficult examiner",hard)+section("Level 4 Trap questions",trap)+"\n## Mock viva\nAnswer one question at a time. Score accuracy, evidence, caveat discipline and clarity; correct it, give ideal answer and ask a harder follow-up.\n"+SRC
H="""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Viva Study Guide</title><style>body{margin:0;background:#f6fafb;color:#152a38;font:16px Arial}aside{position:fixed;width:220px;height:100vh;padding:25px;background:#0d405b;color:white}aside a{display:block;color:#d8ebf3;padding:7px;text-decoration:none}main{margin-left:220px;padding:40px 8%;max-width:1050px}h2{color:#0d405b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}.card,table,details{background:white;border:1px solid #d6e4e8;border-radius:6px}.card{padding:14px;border-left:5px solid #0b8392}.alert{border-left-color:#e75858}.num{font-size:30px;font-weight:bold;color:#0d405b}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #d6e4e8;text-align:left}th{background:#e7f2f5}details{padding:12px;margin:10px 0}@media(max-width:720px){aside{position:static;width:auto;height:auto}main{margin:0}}</style></head><body><aside><h2 style='color:white'>Viva guide</h2><a href=#t>Thesis</a><a href=#d>Data</a><a href=#a>Ablation</a><a href=#c>Calibration</a><a href=#g>Age gap</a><a href=#e>External</a></aside><main><section id=t><h2>What this study is about</h2><p>Aggregate benchmark performance can hide clinically meaningful subgroup failure.</p><div class=grid><div class=card><b>Macro-F1</b><div class=num>.8047</div></div><div class=card><b>ECE</b><div class=num>.1575 to .0206</div></div><div class='card alert'><b>Under 40</b><div class=num>.143</div>3 of 21</div></div></section><section id=d><h2>Data/pipeline</h2><p>HAM10000: 10,015 images, 7,470 lesions. Split by lesion ID: 6,981/1,532/1,502. Six CNN vote -> 24 TTA -> Dirichlet -> diagnosis/referral/conformal/lambda.</p></section><section id=a><h2>Ablation</h2><p>.7459 single CNN -> .7718 ensemble -> .7859 TTA -> .8047 Dirichlet. Only ensemble has clear paired statistical result. Ridge rejected because validation lost.</p></section><section id=c><h2>Calibration</h2><p>Underconfident ensemble. ECE improves, yet sensitivity .7862 to .7310. Calibration is not screening safety.</p></section><section id=g><h2>Hidden age gap</h2><p>Under-40 .143 versus 60+ .764. Lambda total .731 to .831 but young .143 to .238, NNB 3.0 to 6.2. Mitigated, not closed.</p></section><section id=e><h2>External</h2><p>PAD is smartphone shift, not clinical validation. Mechanism claims fail at BCN/MSKCC. Operating point transports; explanation does not.</p></section></main></body></html>"""
readme="# Researcher understanding package\n\n01 Executive; 02 Story; 03 Method; 04 Results; 05 Claim audit; 06 Limitations; 07 Viva bank; 08 Presentation script; 09 Glossary; 10 Quick revision; Viva cheat sheet; One page; 11 Visual plan; interactive HTML."
w("README.md",readme);w("01_EXECUTIVE_SUMMARY.md",E);w("02_RESEARCH_STORY.md",ST);w("03_METHOD_DEEP_DIVE.md",M);w("04_RESULTS_DEEP_DIVE.md",R);w("05_CLAIM_AUDIT.md",C);w("06_LIMITATIONS.md",X);w("07_VIVA_QUESTIONS.md",B);w("08_PRESENTATION_SCRIPT.md",P);w("09_GLOSSARY.md",G);w("10_QUICK_REVISION.md",Q);w("VIVA_CHEATSHEET.md",Q);w("ONE_PAGE_CHEATSHEET.md",Q.split("## If I only have 30 minutes")[0]);w("11_VISUAL_PLAN.md",V);w("viva_study_guide.html",H)
print("created",out)
