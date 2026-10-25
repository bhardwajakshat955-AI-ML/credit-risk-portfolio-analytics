// Builds deliverables/cro_pitch_deck.pptx from the actual pipeline outputs in reports/.
// Run from the project root:  node deliverables/build_cro_deck.js
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const metrics = JSON.parse(fs.readFileSync(path.join(ROOT, "reports/portfolio_metrics.json")));
const model = JSON.parse(fs.readFileSync(path.join(ROOT, "reports/model_results.json")));
const csv = (f) => fs.readFileSync(path.join(ROOT, "reports/tables", f), "utf8").trim().split("\n").map((l) => l.split(","));
const loanType = csv("profile_loan_type.csv");
const incomeBand = csv("profile_income_band.csv");
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
const m = (x) => "$" + (x / 1e6).toFixed(0) + "m";

const THEME = {
  name: "Credit Risk CRO",
  headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F2A44", lt1: "FFFFFF", dk2: "4A5568", lt2: "EEF2F6", accent1: "B23A48", accent2: "4A6FA5",
            accent3: "7FB77E", accent4: "E8B04B", accent5: "9FB3C8", accent6: "6B7B8C", hlink: "4A6FA5", folHlink: "6B7B8C" },
};
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = "Credit Risk Analytics"; pres.title = "Credit Risk Portfolio Analytics — CRO briefing";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK", background: { color: THEME.colors.dk1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.4, w: 11.7, h: 1.4, fontSize: 40, bold: true, color: C.background1, align: "left" } } },
            { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 3.9, w: 11.0, h: 1.6, fontSize: 18, color: C.background2, align: "left" } } }],
});
pres.defineSlideMaster({
  title: "LIGHT", background: { color: THEME.colors.lt1 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.4, w: 12.1, h: 0.9, fontSize: 26, bold: true, color: C.text1, align: "left", margin: 0 } } },
            { text: { text: "Credit Risk Portfolio Analytics · all figures computed from the supplied dataset", options: { x: 0.6, y: 7.05, w: 9, h: 0.3, fontSize: 9, color: C.text2, margin: 0 } } },
            { slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 9, color: C.text2 } }],
});

const card = (s, x, y, w, h, big, label, sub, fill = C.background2) => {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
  s.addText(big, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.8, fontSize: 30, bold: true, color: C.accent1, margin: 0, isTextBox: true });
  s.addText(label, { x: x + 0.2, y: y + 0.92, w: w - 0.4, h: 0.4, fontSize: 13, bold: true, color: C.text1, margin: 0, isTextBox: true });
  if (sub) s.addText(sub, { x: x + 0.2, y: y + 1.3, w: w - 0.4, h: h - 1.4, fontSize: 11, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
};
const bullets = (s, items, x, y, w, h, size = 14) =>
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 6 } })),
            { x, y, w, h, fontSize: size, color: C.text1, valign: "top", margin: 0, isTextBox: true });
const section = (s, text, x, y, w) => s.addText(text, { x, y, w, h: 0.4, fontSize: 16, bold: true, color: C.accent2, margin: 0, isTextBox: true });
const tableOpts = { x: 0.6, fontSize: 12, color: THEME.colors.dk1, border: { type: "solid", pt: 0.5, color: THEME.colors.lt2 }, fill: { color: THEME.colors.lt1 }, autoPage: false, rowH: 0.36 };
const hdr = (t) => ({ text: t, options: { bold: true, fill: { color: THEME.colors.lt2 }, color: THEME.colors.dk1 } });

// ---------------------------------------------------------------- 1. Title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addText("Credit Risk Portfolio Analytics", { placeholder: "title" });
s.addText("Where our default risk sits, how well we can predict it, and what to do about it\nBriefing for the Chief Risk Officer · 10 minutes", { placeholder: "body" });
s.addText(`${metrics.overall.loans.toLocaleString()} loans · ${metrics.overall.customers.toLocaleString()} customers · ${m(metrics.overall.exposure)} originated exposure`,
          { x: 0.8, y: 6.3, w: 11, h: 0.4, fontSize: 14, color: C.accent5, margin: 0, isTextBox: true });
s.addNotes("Agenda: problem, portfolio evidence, model, dashboard and AI feature, governance, actions. Everything shown is computed from the five supplied tables; nothing is illustrative.");

// ---------------------------------------------------------------- 2. Problem
pres.addSection({ title: "Evidence" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Evidence" });
s.addText("One in six loans defaults, and losses are concentrated where we lend most", { placeholder: "title" });
const o = metrics.overall;
card(s, 0.6, 1.5, 2.9, 2.0, pct(o.loan_default_rate), "Loan-level default rate", `${o.loans.toLocaleString()} loan accounts`);
card(s, 3.7, 1.5, 2.9, 2.0, pct(o.customer_default_rate), "Customer-level default rate", "customers with ≥1 defaulted loan; 39% of customers hold 2–3 loans");
card(s, 6.8, 1.5, 2.9, 2.0, pct(o.exposure_weighted_default_rate), "Exposure in default", "originated amount of defaulted loans ÷ total originated amount");
card(s, 9.9, 1.5, 2.8, 2.0, m(o.exposure), "Originated exposure", "sum of loan_amount across the book");
section(s, "Why it matters", 0.6, 3.8, 6);
bullets(s, [
  "Expected loss is driven by frequency × size: the biggest tickets (mortgages) also default most often, so exposure in default (21%) runs well above the loan-count rate (16%).",
  "We currently have no systematic view of which segments carry the risk, no scored watch-list for accounts drifting towards default, and no audited way to explain a score.",
  "This work delivers all three on the supplied data: a cleaned loan-level dataset, a validated scoring model with per-account explanations, and an analyst dashboard with an embedded anomaly feature.",
], 0.6, 4.25, 12.1, 2.6);
s.addNotes("Denominators matter: loan-level vs customer-level vs exposure-weighted rates are all shown separately throughout, and the dashboard labels which is which.");

// ---------------------------------------------------------------- 3. Portfolio overview
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Evidence" });
s.addText("Portfolio overview: mortgages are half the exposure and the worst performer", { placeholder: "title" });
const lt = loanType.slice(1);
s.addChart(pres.ChartType.bar, [
  { name: "Share of exposure (%)", labels: lt.map((r) => r[0]), values: lt.map((r) => +(+r[8] * 100).toFixed(1)) },
  { name: "Loan default rate (%)", labels: lt.map((r) => r[0]), values: lt.map((r) => +(+r[9] * 100).toFixed(1)) },
], { x: 0.6, y: 1.4, w: 7.4, h: 5.3, barDir: "col", barGrouping: "clustered", chartColors: [THEME.colors.accent5, THEME.colors.accent1],
     showLegend: true, legendPos: "b", legendFontFace: "+mn-lt", legendFontSize: 11, showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 10, dataLabelFontFace: "+mn-lt", dataLabelColor: THEME.colors.dk1,
     catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, catAxisLabelColor: THEME.colors.dk2, valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 10, valAxisLabelColor: THEME.colors.dk2,
     valGridLine: { color: THEME.colors.lt2, size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "By loan type", titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: THEME.colors.dk2 });
section(s, "What the profiling shows", 8.4, 1.4, 4.4);
bullets(s, [
  `Mortgages: ${pct(metrics.mortgage.loan_share)} of loans, ${pct(metrics.mortgage.exposure_share)} of exposure, ${pct(metrics.mortgage.default_rate)} default rate vs ${pct(metrics.mortgage.others_default_rate)} for everything else.`,
  `Income gradient is monotonic: ${pct(metrics.income.lt40k_default_rate)} default under $40k vs ${pct(metrics.income.gt100k_default_rate)} above $100k.`,
  `Age: 21–30s are ${pct(metrics.age.young_exposure_share)} of exposure at ${pct(metrics.age.young_default_rate)}; 61+ default at ${pct(metrics.age.old_default_rate)}.`,
  `Region is flat (${pct(metrics.region.min_rate)}–${pct(metrics.region.max_rate)}); channel and city tier differ by under two points.`,
  `Vintages 2018–2023 range ${pct(metrics.vintage.min_rate)}–${pct(metrics.vintage.max_rate)} with no trend; ${metrics.vintage.max_year} weakest.`,
], 8.4, 1.85, 4.4, 4.9, 12);
s.addNotes("Default rates here are loan-level. Exposure = originated loan amount (assumption A6); current outstanding is shown at account level in the dashboard.");

// ---------------------------------------------------------------- 4. Insights
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Evidence" });
s.addText("Highest-impact insights and risk concentrations", { placeholder: "title" });
const mo = metrics.mortgage, inc = metrics.income, un = metrics.unemployed, bu = metrics.bureau;
card(s, 0.6, 1.4, 3.95, 2.75, pct(mo.lt60k_default_rate), "Mortgages to sub-$60k earners",
     `${mo.lt60k_n} loans, ${m(mo.lt60k_exposure)} exposure. Mortgage book overall ${pct(mo.default_rate)}; ${pct(mo.exposure_weighted_default_rate)} of mortgage exposure is in default. Largest concentration = worst performance.`);
card(s, 4.7, 1.4, 3.95, 2.75, "3.4×", "Lowest vs highest income band",
     `<$40k defaults at ${pct(inc.lt40k_default_rate)} vs ${pct(inc.gt100k_default_rate)} for $100k+. Top loan-to-income quintile ${pct(inc.lti_top_quintile_rate)} vs ${pct(inc.lti_bottom_quintile_rate)} bottom. Affordability is the strongest application-stage signal.`);
card(s, 8.8, 1.4, 3.95, 2.75, pct(un.default_rate), "'Unemployed' borrowers",
     `${un.n} loans (${pct(un.exposure_share)} of exposure) yet ${pct(un.defaulted_exposure_share)} of defaulted exposure. ${pct(un.customer_default_rate)} of these customers have a defaulted loan vs ${pct(un.others_customer_default_rate)}. Their recorded income is undefined (open question).`);
card(s, 0.6, 4.35, 6.0, 2.4, pct(bu.wo2_default_rate), "Two or more bureau write-offs",
     `vs ${pct(bu.wo0_default_rate)} with a clean file and ${pct(bu.nobureau_default_rate)} with no bureau file at all (thin file is riskier than clean). Usable at application time, before any repayment is seen.`);
card(s, 6.75, 4.35, 6.0, 2.4, pct(metrics.age.young_lt40k_rate), "Young and low-income",
     `21–30 year-olds under $40k (n=${metrics.age.young_lt40k_n}) vs ${pct(metrics.age.young_gt100k_rate)} for 21–30s over $100k. The age effect is mostly an income and leverage effect — segment by age × income, not age alone.`);
s.addNotes("Each card maps to an insight in reports/portfolio_findings.md with its denominator and sample size. Cells with fewer than 30 loans are never used as evidence.");

// ---------------------------------------------------------------- 5. Modelling approach
pres.addSection({ title: "Model" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Model" });
s.addText("Modelling approach: a behavioural scorecard built without leakage", { placeholder: "title" });
const steps = [["1", "Loan-level dataset", "5 tables joined at loan grain; bureau, repayment and collections aggregated first; row count asserted"],
               ["2", "Engineered features", "12-month repayment behaviour (missed, DPD, trend), bureau snapshot (write-offs, DPD, utilisation), affordability ratios"],
               ["3", "Leakage controls", "Prediction point = end of month 12; collections and gender excluded; customers never split across train/test; preprocessing fit on train only"],
               ["4", "Two models", "Logistic regression baseline and LightGBM; hyper-parameters by customer-grouped CV; out-of-time check on the 2023 vintage"]];
steps.forEach(([n, h, d], i) => {
  const y = 1.45 + i * 1.3;
  s.addShape(pres.ShapeType.ellipse, { x: 0.6, y: y + 0.1, w: 0.6, h: 0.6, fill: { color: C.accent2 }, line: { color: C.accent2 } });
  s.addText(n, { x: 0.6, y: y + 0.1, w: 0.6, h: 0.6, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(h, { x: 1.4, y: y, w: 5.3, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  s.addText(d, { x: 1.4, y: y + 0.4, w: 5.3, h: 0.8, fontSize: 11.5, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
});
const tm = model.test_metrics, oot = model.out_of_time_2023;
section(s, "Held-out test set (20% of customers)", 7.2, 1.45, 5.6);
s.addTable([[hdr("Model"), hdr("ROC-AUC"), hdr("Gini"), hdr("KS")],
            ["Logistic regression", tm.logistic_regression.roc_auc.toFixed(3), tm.logistic_regression.gini.toFixed(3), tm.logistic_regression.ks.toFixed(3)],
            ["LightGBM", tm.lightgbm.roc_auc.toFixed(3), tm.lightgbm.gini.toFixed(3), tm.lightgbm.ks.toFixed(3)]],
           { ...tableOpts, x: 7.2, y: 1.9, w: 5.6, colW: [2.3, 1.1, 1.1, 1.1] });
section(s, "Out-of-time: 2023 vintage", 7.2, 3.25, 5.6);
s.addTable([[hdr("Model"), hdr("ROC-AUC"), hdr("Gini"), hdr("KS")],
            ["Logistic regression", oot.logistic_regression.roc_auc.toFixed(3), oot.logistic_regression.gini.toFixed(3), oot.logistic_regression.ks.toFixed(3)],
            ["LightGBM", oot.lightgbm.roc_auc.toFixed(3), oot.lightgbm.gini.toFixed(3), oot.lightgbm.ks.toFixed(3)]],
           { ...tableOpts, x: 7.2, y: 3.7, w: 5.6, colW: [2.3, 1.1, 1.1, 1.1] });
s.addText("Gini = 2 × AUC − 1; KS = max(TPR − FPR). The two models are statistically tied: the signal in this data is close to additive, so the simpler scorecard is as good as the boosted model.",
          { x: 7.2, y: 5.05, w: 5.6, h: 1.4, fontSize: 11.5, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
s.addNotes("Recommendation in model_evaluation_notes.md: LightGBM powers the dashboard (calibrated probabilities, exact SHAP); logistic regression is the production-scorecard candidate with LightGBM as challenger.");

// ---------------------------------------------------------------- 6. Performance & limitations
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Model" });
s.addText("Model performance: a useful ranking, not an auto-decision engine", { placeholder: "title" });
const seg = model.segment_default_rate_test;
s.addChart(pres.ChartType.bar, [{ name: "Observed default rate (%)", labels: ["Low", "Medium", "High"], values: ["Low", "Medium", "High"].map((k) => +(seg[k].default_rate * 100).toFixed(1)) }],
  { x: 0.6, y: 1.4, w: 5.6, h: 4.0, barDir: "col", chartColors: [THEME.colors.accent3, THEME.colors.accent4, THEME.colors.accent1], showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: THEME.colors.dk1,
    catAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, catAxisLabelColor: THEME.colors.dk2, valAxisLabelFontFace: "+mn-lt", valAxisLabelFontSize: 10, valAxisLabelColor: THEME.colors.dk2,
    valGridLine: { color: THEME.colors.lt2, size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "Observed default rate by model risk segment (test set)", titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: THEME.colors.dk2 });
s.addText(`Segments: High = top 10% of training scores (${seg.High.loans} test loans), Medium = next 20% (${seg.Medium.loans}), Low = rest (${seg.Low.loans}). A 3-to-1 spread in observed default across segments.`,
          { x: 0.6, y: 5.5, w: 5.6, h: 1.2, fontSize: 11.5, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
const th = model.threshold_metrics.lightgbm_high_cutoff, sens = model.sensitivity_lightgbm;
section(s, "What the engineered features add (LightGBM AUC)", 6.6, 1.4, 6.2);
s.addTable([[hdr("Feature set"), hdr("AUC")],
            ["Application fields only", sens.application_only.roc_auc.toFixed(3)],
            ["+ bureau (no repayment)", sens.no_repayment.roc_auc.toFixed(3)],
            ["+ repayment (no bureau)", sens.no_bureau.roc_auc.toFixed(3)],
            ["Full model", tm.lightgbm.roc_auc.toFixed(3)]], { ...tableOpts, x: 6.6, y: 1.85, w: 6.2, colW: [4.4, 1.8] });
section(s, "Limitations to keep in view", 6.6, 3.9, 6.2);
bullets(s, [
  `At the High cut-off, precision is ${pct(th.precision, 0)}: roughly two in three flagged accounts would not default. Thresholds must be set against collections capacity.`,
  "Target definition, prediction timing and the bureau as-of date are assumptions (A1–A3); the time-filtered bureau check costs only 0.001 AUC but cannot rule leakage out.",
  "Age is a driver; an age-free variant must be validated before any production use (A9).",
  "Data is synthetic: metrics describe this dataset, not a real book.",
], 6.6, 4.35, 6.2, 2.5, 11.5);
s.addNotes("Top drivers by SHAP: annual income, years employed, delinquent months in year one, age, mean bureau DPD — consistent with the cross-tabs and the logistic coefficients.");

// ---------------------------------------------------------------- 7. Dashboard
pres.addSection({ title: "Tooling" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Tooling" });
s.addText("Dashboard walkthrough: from the portfolio to a single account", { placeholder: "title" });
const panes = [["Portfolio view", ["KPIs with explicit denominators", "Risk-segment mix and observed default", "Exposure & default by income, employment, region, age, loan type", "Vintage trend, cross-cut heatmap, full-book insights"]],
               ["Working filters", ["Region · income band · employment type · loan type · risk segment", "Every metric, chart and the account list recompute on selection", "Empty selections and small cells handled explicitly"]],
               ["Account view", ["Account and customer IDs only — no personal detail", "PD estimate (labelled as an estimate), risk segment, in-/out-of-sample note", "Top-8 SHAP drivers with the feature's value", "12-month repayment chart, bureau lines, collections events, anomaly flags"]]];
panes.forEach(([h, items], i) => {
  const x = 0.6 + i * 4.15;
  s.addShape(pres.ShapeType.roundRect, { x, y: 1.4, w: 3.95, h: 3.6, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08 });
  s.addText(h, { x: x + 0.2, y: 1.55, w: 3.6, h: 0.4, fontSize: 15, bold: true, color: C.text1, margin: 0, isTextBox: true });
  bullets(s, items, x + 0.2, 2.0, 3.6, 2.9, 11.5);
});
section(s, "Account-level use case (ACC200001, a $306k mortgage)", 0.6, 5.2, 12);
s.addText("The analyst opens the account from the High-risk list: estimated PD 31% (segment High, portfolio base rate 16%), four late months with a maximum of 14 days past due, no bureau file, and drivers showing a high loan-to-income ratio on a 360-month loan, the delinquent months, the loan size and a short employment tenure. The analyst can see every number behind the score, record a review decision, and route the account to early engagement — or dismiss it with a note. The score is decision support; the analyst decides.",
          { x: 0.6, y: 5.6, w: 12.1, h: 1.3, fontSize: 12, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
s.addNotes("Live demo: streamlit run app/dashboard.py. All four pages were exercised by automated tests (filters, invalid IDs, thin-file customers, empty selections).");

// ---------------------------------------------------------------- 8. AI feature
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Tooling" });
s.addText("Embedded AI feature: anomaly detection with a human in the loop", { placeholder: "title" });
const rules = [["R1", "DPD spike vs own history", "Months 10–12 mean DPD ≥ 20 days above months 1–9, worst recent month ≥ 30 DPD"],
               ["R2", "Missed payments vs peers", "Missed months above the 95th percentile of loan-type × income-band peers (≥ 2)"],
               ["R3", "Underpayment vs peers", "Paid/due ratio below the peer 5th percentile"],
               ["R4", "External stress hidden", "≥ 2 bureau write-offs or bureau DPD ≥ 60 while internal max DPD ≤ 15"]];
s.addTable([[hdr("Rule"), hdr("Pattern"), hdr("Trigger")], ...rules.map((r) => [r[0], r[1], r[2]])],
           { ...tableOpts, x: 0.6, y: 1.4, w: 7.4, colW: [0.7, 2.3, 4.4], fontSize: 11 });
s.addText("Across the book, 1,420 accounts (9.7%) carry at least one flag; 351 carry two or more. Flagged accounts defaulted at 23% vs 15% for unflagged — enriched for risk, but a flag is not a default and not a fraud finding. R4 accounts defaulted at the base rate: it is a watch-list rule, not a predictor.",
          { x: 0.6, y: 3.75, w: 7.4, h: 1.3, fontSize: 11.5, color: C.text2, margin: 0, isTextBox: true, valign: "top" });
section(s, "Human review before any action", 0.6, 5.1, 7.4);
["Flag raised with its evidence, threshold and comparison group", "Analyst reviews repayment chart, bureau and model score in the dashboard", "Decision recorded: confirm → collections team; dismiss → note kept; needs information", "Nothing is contacted, cut or reported automatically"].forEach((t, i) => {
  s.addShape(pres.ShapeType.ellipse, { x: 0.6 + i * 1.85, y: 5.55, w: 0.4, h: 0.4, fill: { color: C.accent1 }, line: { color: C.accent1 } });
  s.addText(String(i + 1), { x: 0.6 + i * 1.85, y: 5.55, w: 0.4, h: 0.4, fontSize: 12, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(t, { x: 0.6 + i * 1.85, y: 6.0, w: 1.75, h: 0.95, fontSize: 9.5, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
});
s.addShape(pres.ShapeType.roundRect, { x: 8.4, y: 1.4, w: 4.4, h: 5.55, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08 });
s.addText("False positive", { x: 8.6, y: 1.55, w: 4.0, h: 0.35, fontSize: 14, bold: true, color: C.accent1, margin: 0, isTextBox: true });
s.addText("Two instalments paid 25 days late around a job change, now current (R1); a negotiated lower instalment (R3); a settled write-off the bureau has not updated (R4). Cost: analyst time, or an unjustified contact if review is skipped.",
          { x: 8.6, y: 1.9, w: 4.0, h: 1.5, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
s.addText("False negative", { x: 8.6, y: 3.5, w: 4.0, h: 0.35, fontSize: 14, bold: true, color: C.accent1, margin: 0, isTextBox: true });
s.addText("A borrower late every month from month 1 shows no spike and may sit under the peer cut; distress with a lender that does not report to the bureau is invisible. The feature complements the score; it does not replace it.",
          { x: 8.6, y: 3.85, w: 4.0, h: 1.5, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
s.addText("Safeguards", { x: 8.6, y: 5.45, w: 4.0, h: 0.35, fontSize: 14, bold: true, color: C.accent1, margin: 0, isTextBox: true });
s.addText("Deterministic rules, fixed and documented thresholds, evidence shown for every flag, review queue exportable for audit, no automated action.",
          { x: 8.6, y: 5.8, w: 4.0, h: 1.1, fontSize: 11, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
s.addNotes("Option A (anomaly detection) was chosen over 'Ask the portfolio' because it can be implemented fully and verifiably with the supplied data, with no external API dependency.");

// ---------------------------------------------------------------- 9. Governance
pres.addSection({ title: "Governance" });
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Governance" });
s.addText("Risk, compliance, fairness, explainability and governance", { placeholder: "title" });
const gov = [["Responsible use", "Scores and flags are decision support with mandatory human review; no automated adverse action. The risk of acting on a wrong score is addressed by precision figures on every operating point and by the review queue."],
             ["Data privacy", "Dashboard shows account and customer IDs only; gender excluded from model and views. Production needs SSO, role-based access, audit logs and a data-protection assessment."],
             ["Explainability", "Per-account SHAP contributions and a transparent logistic baseline; feature dictionary generated from code so documentation cannot drift."],
             ["Fairness", "Gender disparity check shows near-identical mean PD and High-risk share (0.159 vs 0.164; 9% vs 11%). Age is a driver and a protected characteristic — needs a compliance ruling and an age-free variant."],
             ["Model validation", "Customer-grouped hold-out plus out-of-time vintage test; both agree. Independent validation, PSI/CSI monitoring, retraining and rollback are designed but not yet operated."],
             ["Auditability", "Fixed seeds, reproducible pipeline, versioned artefacts, assumptions log with 14 assumptions and 7 open questions. Regulatory compliance has not been assessed and is not claimed."]];
gov.forEach(([h, d], i) => {
  const col = i % 2, row = Math.floor(i / 2), x = 0.6 + col * 6.15, y = 1.4 + row * 1.85;
  s.addShape(pres.ShapeType.roundRect, { x, y, w: 5.95, h: 1.7, fill: { color: C.background2 }, line: { color: C.background2 }, rectRadius: 0.08 });
  s.addText(h, { x: x + 0.2, y: y + 0.12, w: 5.55, h: 0.35, fontSize: 14, bold: true, color: C.accent2, margin: 0, isTextBox: true });
  s.addText(d, { x: x + 0.2, y: y + 0.5, w: 5.55, h: 1.15, fontSize: 10.5, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
});
s.addNotes("Nothing here claims compliance. The governance items are the prerequisites listed in the assumptions log, section C.");

// ---------------------------------------------------------------- 10. Actions & roadmap
s = pres.addSlide({ masterName: "LIGHT", sectionTitle: "Governance" });
s.addText("Recommended actions, expected value and the road to production", { placeholder: "title" });
section(s, "Actions the risk team can take now", 0.6, 1.4, 6.0);
bullets(s, [
  "Review mortgage affordability: income floor or loan-to-income cap for sub-$60k applicants; track mortgage exposure share as a concentration metric.",
  "Add policy rules ahead of the model: two or more bureau write-offs → manual review; 'Unemployed' with undefined income → documented income or secured-only.",
  "Treat thin-file applicants as their own policy segment (lower initial limits).",
  "Use the High segment (top 10%) as an early-engagement list and the anomaly queue as a weekly analyst review; set thresholds against collections capacity.",
], 0.6, 1.85, 6.0, 3.0, 11.5);
section(s, "Expected value — hypotheses, not measured outcomes", 0.6, 4.9, 6.0);
bullets(s, [
  "Concentrating early engagement on the High segment reaches 23% of defaulters from 10% of accounts (test set); any loss reduction depends on collections effectiveness, which this data cannot measure.",
  "Policy rules on write-offs and affordability would have touched the segments with 2–3× the book default rate; impact on volume must be weighed by the business.",
], 0.6, 5.35, 6.0, 1.6, 11);
section(s, "Roadmap and what we need from the bank", 7.0, 1.4, 5.8);
[["Weeks 1–4", "Confirm default definition, prediction timing, income semantics, bureau as-of date (Q1–Q7); fix collections dates."],
 ["Weeks 5–8", "Rebuild features on point-in-time data; validate age-free variant; compliance ruling; independent validation of the scorecard."],
 ["Weeks 9–12", "Deploy behind SSO with audit logging; PSI/CSI monitoring; analyst pilot of the anomaly queue; tune thresholds on reviewed outcomes."],
 ["Ongoing", "Quarterly performance review; annual or event-triggered retraining with validation and rollback."]].forEach(([w, d], i) => {
  const y = 1.9 + i * 1.22;
  s.addShape(pres.ShapeType.roundRect, { x: 7.0, y, w: 1.45, h: 1.05, fill: { color: C.accent2 }, line: { color: C.accent2 }, rectRadius: 0.06 });
  s.addText(w, { x: 7.0, y, w: 1.45, h: 1.05, fontSize: 11, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0, isTextBox: true });
  s.addText(d, { x: 8.6, y: y + 0.05, w: 4.2, h: 1.0, fontSize: 10.5, color: C.text1, margin: 0, isTextBox: true, valign: "top" });
});
s.addNotes("Open assumptions and questions are in reports/assumptions_log.md. The deck ends with a decision request: approve the discovery phase (weeks 1–4) and the compliance review of age as a feature.");

(async () => {
  const out = path.join(__dirname, "cro_pitch_deck.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
