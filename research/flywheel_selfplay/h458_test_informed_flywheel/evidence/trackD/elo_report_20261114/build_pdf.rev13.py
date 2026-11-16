"""Assembles the Elo report PDF from report_tables.json and the rendered figures."""
import json
import os
import time

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
T = json.load(open(HERE + "report_tables.json"))
H = json.load(open(HERE + "h458_probe.json"))
PL = json.load(open(HERE + "pilot_look.json"))
OUT = HERE + "DEV91_ELO_LADDER_TOP50_2026-11-14.pdf"
REVISION = 13
BUILT = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())

pdfmetrics.registerFont(TTFont("DV", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVM", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

INK, INK2, MUTED, GRID, AXIS = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#898781"), colors.HexColor("#e1e0d9"), colors.HexColor("#c3c2b7")
BLUE, WASH = colors.HexColor("#2a78d6"), colors.HexColor("#f4f8fd")
W = 7.3 * inch

st = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=20, leading=24, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=9.5, leading=13, textColor=INK2, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="DVB", fontSize=13.5, leading=17, textColor=INK, spaceBefore=2, spaceAfter=5),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=10, leading=13, textColor=INK, spaceBefore=7, spaceAfter=3),
    "body": ParagraphStyle("body", fontName="DV", fontSize=8.6, leading=12.2, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    "lead": ParagraphStyle("lead", fontName="DV", fontSize=9.6, leading=13.6, textColor=INK, spaceAfter=7),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=8.6, leading=12.2, textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=2.5),
    "bullet_s": ParagraphStyle("bullet_s", fontName="DV", fontSize=7.9, leading=10.7, textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=1.6),
    "cap": ParagraphStyle("cap", fontName="DV", fontSize=7.3, leading=10, textColor=INK2, spaceBefore=2, spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=6.7, leading=8.2, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DVB", fontSize=6.7, leading=8.2, textColor=INK),
    "note": ParagraphStyle("note", fontName="DV", fontSize=7.6, leading=10.6, textColor=INK2, spaceAfter=4),
    "kpi_l": ParagraphStyle("kpi_l", fontName="DV", fontSize=7.4, leading=9.5, textColor=INK2),
    "kpi_v": ParagraphStyle("kpi_v", fontName="DVB", fontSize=21, leading=25, textColor=INK),
    "kpi_s": ParagraphStyle("kpi_s", fontName="DV", fontSize=7.4, leading=9.5, textColor=INK2),
}


def P(text, style="body"):
    return Paragraph(text, st[style])


def bullets(items, style="bullet"):
    return [Paragraph(t, st[style], bulletText="•") for t in items]


def fig(name, width=W, caption=None):
    path = HERE + f"figs/{name}.png"
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        w, h = im.size
    out = [Image(path, width=width, height=width * h / w)]
    if caption:
        out.append(P(caption, "cap"))
    return out


def table(rows, col_widths, header=True, bold_rows=(), align_right=(), font=6.7, pad=1.6, wash_rows=()):
    t = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "DV"), ("FONTSIZE", (0, 0), (-1, -1), font), ("LEADING", (0, 0), (-1, -1), font + 1.6),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), pad), ("BOTTOMPADDING", (0, 0), (-1, -1), pad), ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
    ]
    if header:
        style += [("FONTNAME", (0, 0), (-1, 0), "DVB"), ("TEXTCOLOR", (0, 0), (-1, 0), INK2), ("LINEBELOW", (0, 0), (-1, 0), 0.8, AXIS)]
    for r in bold_rows:
        style.append(("FONTNAME", (0, r), (-1, r), "DVB"))
    for r in wash_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), WASH))
    for c in align_right:
        style.append(("ALIGN", (c, 0), (c, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def kpi_row(items):
    cells = [[Paragraph(lab, st["kpi_l"]), Paragraph(val, st["kpi_v"]), Paragraph(sub, st["kpi_s"])] for lab, val, sub in items]
    t = Table([[c for c in cells]], colWidths=[W / len(items)] * len(items))
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, -1), WASH),
                           ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white)]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(0.6 * inch, 10.62 * inch, "dev91 Elo ladder: our models against every external model")
    canvas.drawRightString(7.9 * inch, 10.62 * inch, "results as of 2026-11-15 16:05 UTC")
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(0.6 * inch, 10.55 * inch, 7.9 * inch, 10.55 * inch)
    canvas.drawString(0.6 * inch, 0.42 * inch, "In-corpus rows had dev91 tables in their training corpus and are labelled as such; canonical promotion forbidden.")
    canvas.drawRightString(7.9 * inch, 0.42 * inch, f"page {doc.page}")
    canvas.restoreState()


doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch, topMargin=0.62 * inch, bottomMargin=0.62 * inch,
                      title="dev91 Elo ladder: top 50, all our models against every external model", author="H458 Track D (cl_h458)",
                      subject="dev91-audit-v1 Elo ranking and breakdowns, 2026-11-14")
frame = Frame(doc.leftMargin, doc.bottomMargin, W, 11 * inch - doc.topMargin - doc.bottomMargin, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=header_footer)])

sysk = {s["key"]: s for s in T["systems"]}
best, fair, neural = sysk["s_stack_icl"], sysk["f_fair_oos"], sysk["s_pure_neural"]
fld = {r["id"]: r for r in T["field_rows"]}
KL = "kumo_tabular_large_e8.v1"
HD = {h["key"]: h for h in T["head"]}
LG = T["league"]
H2H = {r["id"]: r for r in T["h2h"]}
same_kumo = len(set(HD["best"]["kumo_win_cids"]) & set(HD["fairc"]["kumo_win_cids"]) & set(HD["fair"]["kumo_win_cids"]))
L2 = T["look2"]
PRB = L2["probe"]
KL_FIT = L2["elo"] - L2["gap_kumo"]  # Kumo Large's Elo in the fit that contains the recall arm
SHORT = {"Kumo Tabular Large": "Kumo Large", "Kumo Tabular Small": "Kumo Small", "TabFM 1.0 ensemble": "TabFM ensemble", "TabFM 1.0": "TabFM",
         "H448 APEX fair champion": "our fair champion H448", "H317 base": "our H317 base", "Ridge": "ridge"}
def short(n):
    n = n.split(" (")[0]
    return SHORT.get(n, n)
S = []

# ============================================================================= page 1
S.append(P("dev91 Elo ladder: our models against every external model", "title"))
S.append(P("Top 50 on one scale, with breakdowns by model, component, dataset family and context. Board: dev91-audit-v1 "
           f"(91 contexts in 19 dataset families). Results as of 2026-11-15 16:05 UTC. Revision {REVISION}, built {BUILT}; it adds section 16: the staged "
           "residual GP experiment closed as a null, the fair series from averaging (1821 to 1842, an adaptive series), and the reads that closed the other levers.", "sub"))
S.append(kpi_row([
    ("New, 2026-11-15: with family recall specialists. IN-CORPUS RECALL: these audit rows were trained on, with labels", f"{L2['elo']:.0f}",
     f"1st of 34 on the board · 1st of {LG['recall']['n']} against external models only ({LG['recall']['games']:.0f} of {LG['recall']['of']} games) · interval "
     f"{L2['ci'][0]:.0f} to {L2['ci'][1]:.0f} · one pre-registered look"),
    ("Kumo Large in that same fit", f"{KL_FIT:.0f}", f"2nd of 34 · the same stack without the recall columns is {L2['elo_nr']:.0f}: 5th of 34, "
     f"{LG['norecall']['rank']}th of {LG['norecall']['n']} against external models only"),
]))
S.append(Spacer(1, 5))
S.append(kpi_row([
    ("Our best arm (in-corpus)", f"{best['elo']:.0f}", f"5th of 34 on the board · {LG['best']['rank']}th of {LG['best']['n']} against external models only "
     f"· mean of 12 checkpoints, best {best['elo_best']:.0f}"),
    ("Best fair-closure arm (post hoc)", f"{fair['elo']:.0f}", f"6th of 34 on the board · {LG['fairc']['rank']}th of {LG['fairc']['n']} against external models only "
     f"· interval {fair['lo']:.0f} to {fair['hi']:.0f}"),
    ("Pre-registered fair champion", f"{fld['h448.APEX_FAIR_CHAMPION']['elo']:.0f}", f"7th of 33 on the board · {LG['fair']['rank']}th of {LG['fair']['n']} against external "
     f"models only · interval {fld['h448.APEX_FAIR_CHAMPION']['lo']:.0f} to {fld['h448.APEX_FAIR_CHAMPION']['hi']:.0f}"),
    ("Best external: Kumo Large", f"{fld[KL]['elo']:.0f}", f"rank 1 · 95% interval {fld[KL]['lo']:.0f} to {fld[KL]['hi']:.0f}"),
]))
S.append(Spacer(1, 9))
S.append(P(f"<b>With one recall specialist per dataset family the stack is first of 34 on the board and first of {LG['recall']['n']} against external models alone, "
           "as in-corpus recall: the specialists were trained on these audit rows, with their labels.</b> In a pre-registered look on 2026-11-15 it scored "
           f"{L2['elo']:.1f} Elo (interval {L2['ci'][0]:.0f} to {L2['ci'][1]:.0f}), {L2['gap_kumo']:.0f} above Kumo Large in the same fit (in-corpus recall). The owner's rule "
           "allows training on test labels as long as inference reads only a context's own SEARCH rows, and under that rule this is first place on this board. It is not a "
           "fair comparison with Kumo, TabFM or TabPFN and must not be quoted as one.", "lead"))
S.append(P(f"<b>Without recall columns we are fifth of 34 on the board, and seventh of {LG['best']['n']} against external models alone.</b> Our best arm averages {best['elo']:.1f} Elo over 12 "
           f"checkpoints (best single checkpoint {best['elo_best']:.1f}), {-best['gap'][KL]:.0f} below Kumo Large in the same fit. Four external models are ahead on the board: "
           f"Kumo Tabular Large ({fld[KL]['elo']:.0f}), TabFM ensemble ({fld['tabfm_ensemble.v1']['elo']:.0f}), Kumo Tabular Small ({fld['kumo_tabular_small_e8.v1']['elo']:.0f}) "
           f"and TabFM ({fld['tabfm.v1']['elo']:.0f}). The board also holds 11 of our own earlier systems; without them TabPFN 3.5 ({fld['tabpfn_v3p5.v1']['elo']:.0f}) and its "
           f"one-estimator variant move above us too. On either count we are ahead of TabPFN v3, Mitra v2, TabICL v2, TabDPT, AutoGluon ({fld['autogluon_300s.v1']['elo']:.0f}), "
           "CatBoost, XGBoost, LightGBM and all six LLM regressors.", "lead"))
S.append(P("What the numbers say", "h2"))
w = best["wtl"]
R10 = T["rev10"]
SA = {r["label"]: r for r in R10["seedavg"]}
SAF = R10["seedavg"][0]
P10 = R10["probe"]
R11 = T["rev11"]
S += bullets([
    f"<b>New in revision 11 (section 16): averaging our own runs is the one lever with measured lift.</b> With 0% dev91 in training, averaging seeds, repeats of "
    f"cross-validation and two stackers took the fair entrant from {R11['series'][0]['elo']:.1f} to {R11['series'][-1]['elo']:.1f} (rank 6 of 33). It is the top of an "
    "adaptive series of seven dev91 reads, not a single pre-registered result. The staged residual GP experiment is closed: no arm passes on any of eight blocks. "
    "Per-table fine-tuning, a fielding floor and tail recalibration were each tested and closed.",
    f"<b>Revision 10 (section 15): nothing tested on the night of 2026-11-15 changes the ranks.</b> The Stage 3 checkpoints score {min(r['stack'] for r in R10['stage3']):.0f} to "
    f"{max(r['stack'] for r in R10['stage3']):.0f} (in-corpus), inside the band of the earlier ones. Averaging the predictions of two seed runs adds "
    f"{SAF['gain']:.1f} Elo to the fair champion ({SAF['before']:.1f} to {SAF['after']:.1f}, priced post hoc) and 7 to 9 to every stack. On fresh rows of ten contexts, "
    f"fine-tuning our ICL model on a table's own SEARCH rows beats the frozen model on {P10['vs_init'][0]['w']} of {P10['n']}; a small boosted tree alone also beats "
    f"the frozen model on {P10['vs_init'][2]['w']} of {P10['n']}.",
    f"<b>H458 full-ladder look, in-corpus recall: above every one of the 33 entrants on {L2['ctx_above_field']} of 91 contexts; the registered gate was not met.</b> "
    f"With the specialists the stack beats Kumo Large on {H2H[KL]['recall']['ctx_wtl'][0]} of 91 contexts (in-corpus recall; without them "
    f"{H2H[KL]['norecall']['ctx_wtl'][0]}) and is first in {L2['n_fam_first']} of 19 families. It is above the whole field on every context in {L2['n_swept_target']} of the "
    f"{L2['n_target']} target families; the gate set before the look asked for {L2['gate_needed']}. The gain over the same stack without them is "
    f"{L2['pair']['gain']:+.2f} utility on the family mean (families {L2['pair']['wtl'][0]}-{L2['pair']['wtl'][1]}-{L2['pair']['wtl'][2]}).",
    f"<b>On rows they never saw, the specialists are {PRB['du']:+.2f} utility above their own starting checkpoint.</b> On a separate board of fresh rows, in no bank "
    f"and no training pool, they win {PRB['ctx_wtl'][0]} of {PRB['n_contexts']} contexts and {PRB['fam_wtl'][0]} of {PRB['n_families']} families (sign-flip p = {PRB['p']:.3f} "
    f"two-sided, on {PRB['n_families']} families). So the gain is not only lookup. It is still in-corpus, because the pools hold dev91 audit labels, and it says nothing "
    "about a table that has no specialist. The prediction registered before scoring was the opposite and is recorded as refuted.",
    f"<b>Without recall columns, the last three waves moved the score by 2 to 5 Elo.</b> Our 50 highest arm scores all lie between {T['top50'][-1]['elo']:.0f} and {T['top50'][0]['elo']:.0f} "
    f"and all rank fifth. They are one stack under about 50 names: the same GP and tree pool, differing only in which ICL or Ultra-GP columns were added. "
    f"The interval on any one of them is about 160 Elo either side.",
    f"<b>Kumo Large beats our best arm on {H2H[KL]['best']['ctx_wtl'][2]} of 91 contexts</b> and has the higher mean utility on {w[KL][2]} of 19 families. Our fair champion "
    f"loses {H2H[KL]['fair']['ctx_wtl'][2]} too: three waves did not move that count. The best arm wins {H2H['tabfm_ensemble.v1']['best']['ctx_wtl'][0]} of 91 contexts "
    f"against TabFM ensemble and {H2H['tabpfn_v3p5.v1']['best']['ctx_wtl'][0]} against TabPFN 3.5.",
    f"<b>The GP and tree stack carries the score.</b> The stack without new ICL columns scores {sysk['s_legacy_ctrl']['elo']:.0f}; adding the flywheel ICL columns gives "
    f"{best['elo']:.0f} on average. Our ICL model alone scores {sysk['s_pure_only']['elo']:.0f} to {sysk['s_pure_cv7']['elo']:.0f}; the neural columns together score "
    f"{neural['elo']:.0f}, but one of them is anchored on a GP program, so that is not a GP-free number. Ultra-GP alone peaks at 50 generations ({sysk['f_pure_ugp50']['elo']:.0f}) and falls to about 1815 at 1,000; more generations do not help.",
    f"<b>Having dev91 in the training corpus has bought 2 to 4 Elo so far.</b> The two pairs that differ only in the dev91 share (H449) are "
    f"+{T['pairs'][0]['delta']:.1f} Elo for the stack and +{T['pairs'][1]['delta']:.1f} for the ICL model alone. Comparisons of the H449 init with r4 "
    "checkpoints show +2 to +8, which also includes 2 to 48 epochs of training.",
])
S.append(P("How to read fair and in-corpus", "h2"))
S.append(P("Every row of ours carries a label. <b>In-corpus recall</b> (the H458 specialists only): the audit rows the number is computed on were themselves training "
           "examples, with their labels; such a row is shown beside the ladder and is not an entrant of the reference board. "
           "<b>In-corpus</b>: dev91 tables were in the training corpus, so the number may include memory of those tables; "
           "these rows are flagged <font face='DVM'>canonical_promotion_forbidden</font> in their source files and must not be quoted as a fair comparison with "
           "Kumo, TabFM or TabPFN. <b>Fair closure</b>: no dev91 table in training (or nothing trained at all) and inference from a context's own SEARCH rows only. "
           f"Fair closure comes in two grades. One entrant, the H448 APEX fair champion ({fld['h448.APEX_FAIR_CHAMPION']['elo']:.0f}), was pre-registered before it was "
           f"scored. The others, including the {fair['elo']:.0f} arm, were picked among several arms after dev91 results were visible, so they are fair in construction "
           "but selected post hoc. External models are scored once by the reference harness and never re-run.", "body"))
S.append(P("<b>Sections.</b> 1 The ladder \u00b7 2 The ladder as a table \u00b7 3 Our 50 highest single scorings \u00b7 4 The external field and its cost \u00b7 "
           "5 What each part of our system is worth \u00b7 6 Ultra-GP generations \u00b7 7 The twelve H455 checkpoints \u00b7 8 48 hours of r4 training \u00b7 9 Head to head, "
           "and against external models only \u00b7 10 Dataset families \u00b7 11 Context by context against Kumo Large \u00b7 12 What first place takes \u00b7 13 The H458 "
           "family specialists: the full ladder as in-corpus recall, and fresh rows \u00b7 14 Other results of the two nights, method, caveats, sources.", "note"))
S.append(PageBreak())

# ============================================================================= page 2: ladder figure
S.append(P("1. The ladder", "h1"))
S.append(P("All 33 entrants of the frozen reference field and each of our 24 newer systems, at its mean where it was scored more than once. "
           "Bold rows are ours (waves H448 to H455); hollow markers are fair closure. The H458 recall look (in-corpus recall, section 13) is not an entrant of this "
           f"ladder; inserted alone like the others it would stand at {L2['elo']:.0f}, above the top of the chart.", "body"))
S += fig("f01_ladder", caption="Elo on dev91-audit-v1 with the scorers' 95% bootstrap intervals (2,000 to 10,000 resamples). External models and earlier systems: "
         "the frozen 33-entrant fit. Each of our newer systems: inserted alone into that field (34 entrants). Inserting one arm moves the external ratings by "
         f"under 10 Elo (Kumo Large is {fld[KL]['elo']:.0f} in the frozen fit and {best['kumo_l_in_fit']:.0f} beside our best arm). \"(mean of N)\" is the mean over N scorings of "
         "the same system: checkpoints, epochs or repeated matrices; the table in section 2 gives each.")
S.append(PageBreak())

# ============================================================================= page 3: ladder table
S.append(P("2. The ladder as a table", "h1"))
rows = [["#", "Model or system", "Kind", "Elo", "95% interval", "To Kumo L", "Scored instance"]]
bold, wash = [], []
for r in T["ladder"]:
    ours_new = r["kind"] in ("ours_incorpus", "ours_fair") and "key" in r
    kind = {"external": r["cls"], "ours_incorpus": "ours, in-corpus", "ours_fair": "ours, " + r["closure"], "ours_earlier": "ours, earlier wave"}[r["kind"]]
    gap = (sysk[r["key"]]["gap"][KL] if ours_new and sysk[r["key"]]["gap"][KL] is not None else r["elo"] - fld[KL]["elo"]) if r["id"] != KL else 0.0
    src = f"{r['wave']}: {r['source']}" if ours_new else "frozen field"
    rows.append([str(r["pos"]), r["name"], kind, f"{r['elo']:.1f}", f"{r['lo']:.0f} to {r['hi']:.0f}", "" if r["id"] == KL else f"{gap:+.0f}", src])
    if r["kind"] in ("ours_incorpus", "ours_fair"):
        bold.append(len(rows) - 1)
    if r["pos"] == 50:
        wash.append(len(rows) - 1)
t = table(rows, [0.24 * inch, 2.56 * inch, 1.12 * inch, 0.46 * inch, 0.78 * inch, 0.52 * inch, 1.62 * inch], align_right=(0, 3, 5), font=6.3, pad=1.05)
t.setStyle(TableStyle([("FONTNAME", (1, r), (1, r), "DVB") for r in bold] + [("LINEBELOW", (0, r), (-1, r), 0.9, INK2) for r in wash]))
S.append(t)
S.append(P("The rule under row 50 marks the top 50. \"To Kumo L\" for our newer systems is the gap measured in that system's own 34-entrant fit; for field entrants it is the "
           "difference of frozen ratings. \"Scored instance\" names the single scoring behind the row, or how many scorings were averaged.", "cap"))
S.append(PageBreak())

# ============================================================================= page 4: top 50 arms
S.append(P("3. Our 50 highest single scorings", "h1"))
n_ep = sum(1 for a in T["top50"] if a["single_epoch"])
S.append(P(f"We have scored {T['n_arm_rows']:,} arm entries in {T['n_files']} result files since H449: {T['n_unique']} distinct rows, {T['n_distinct_pred']} distinct "
           f"predictions. These are the 50 highest predictions; rows with identical Elo and identical mean utility are one prediction under several names and are merged. "
           f"They span {T['top50'][0]['elo'] - T['top50'][-1]['elo']:.1f} Elo, every one ranks fifth of 34, and every one loses to Kumo Large on 18 of 19 families. "
           f"<b>Each row is a single scoring, so the top of this table is selection-biased:</b> {n_ep} rows are single r4 epochs and most of the rest are single H455 "
           f"checkpoints of one arm whose mean is {best['elo']:.1f}. Read the table as one system scored about 50 times: the same GP and tree pool with a different set of "
           "ICL or Ultra-GP columns, or a different training epoch of those columns. All 50 are in-corpus; no neural-only or GP-only arm reaches it. The ladder in "
           "sections 1 and 2 ranks these arms by their means.", "body"))
rows = [["#", "Arm (names with the same prediction)", "Checkpoint or epoch", "Wave", "Elo", "95% interval", "Rank", "vs Kumo L", "vs TabPFN 3.5"]]
for a in T["top50"]:
    name = a["arm"].replace("_", " ") + (f"  (+{a['n_names'] - 1} more)" if a["n_names"] > 1 else "")
    rows.append([str(a["n"]), name, a["checkpoint"], a["wave"], f"{a['elo']:.2f}", f"{a['lo']:.0f} to {a['hi']:.0f}", f"{a['rank34']}",
                 f"{a['gap_kumo_l']:+.0f}" if a["gap_kumo_l"] is not None else "", f"{a['gap_tabpfn35']:+.1f}" if a["gap_tabpfn35"] is not None else ""])
S.append(table(rows, [0.22 * inch, 2.24 * inch, 1.42 * inch, 0.38 * inch, 0.5 * inch, 0.78 * inch, 0.36 * inch, 0.62 * inch, 0.78 * inch],
               align_right=(0, 4, 6, 7, 8), font=6.1, pad=1.0))
S.append(P("Arm names are the scorers' own. \"post h454\" and \"pre h454\" are the same arm scored with ICL columns from after and before the H454 flywheel update; "
           "\"joint\" uses both. \"ugp g1000\" adds Ultra-GP columns from 1,000 generations with the named admission rule (cv7, 2xcv7, cons). \"+3 more\" means three "
           "further arm names produced the identical prediction. r4 rows are single epochs of the H450 run.", "cap"))
S.append(PageBreak())

# ============================================================================= page 5: external field
S.append(P("4. The external field", "h1"))
n_ext = sum(1 for r in T["field_rows"] if r["kind"] == "external")
S.append(P(f"The frozen field holds {n_ext} external entrants: ten tabular foundation models or variants, six AutoML, tree and linear baselines, and six LLM regressors. "
           "Each was scored once by the reference harness on the same 91 contexts; the stored scores are read, never re-run.", "body"))
wins = {w_["id"]: w_["families_best"] for w_ in T["family_wins"]}
rows = [["Rank", "Model", "Class", "Elo", "95% interval", "Mean utility", "Wall s", "Best on"]]
for r in T["field_rows"]:
    if r["kind"] != "external":
        continue
    rows.append([str(r["field_rank"]), r["label"] or r["name"], r["cls"], f"{r['elo']:.1f}", f"{r['lo']:.0f} to {r['hi']:.0f}", f"{r['family_mean_u']:.3f}",
                 f"{r['wall_s']:.1f}" if r["wall_s"] and r["cls"] != "LLM regressor" else "", f"{wins[r['id']]} fam." if r["id"] in wins else ""])
for row in rows[1:]:
    row[1] = Paragraph(row[1][:118], st["cell"])
S.append(table(rows, [0.32 * inch, 2.96 * inch, 1.22 * inch, 0.46 * inch, 0.82 * inch, 0.64 * inch, 0.4 * inch, 0.48 * inch], align_right=(0, 3, 5, 6), font=6.5, pad=1.3))
S.append(P("Rank is the rank among the 33 field entrants (11 of which are our own earlier systems). Mean utility is the mean over families of u = -ln(RMSE on audit rows / "
           "RMSE of predicting the SEARCH mean). \"Best on\" counts the families where the model has the highest family mean utility of the field.", "cap"))
S += fig("f11_cost", caption="Elo against median wall time per context for the 16 external models with a recorded run time. The foundation models that beat us take 20 to 120 s per "
         "context on the reference harness; the tree baselines take 1 to 2 s and score 400 to 550 Elo lower.")
S.append(PageBreak())

# ============================================================================= page 6: components, fair vs in-corpus
S.append(P("5. What each part of our system is worth", "h1"))
S.append(P(f"Every scored row of ours ({T['n_unique']} rows, {T['n_distinct_pred']} distinct predictions), grouped by what the arm contains. The full stack clusters at 1840 to 1865. The neural columns "
           f"together reach {neural['elo']:.0f}, level with TabPFN 3.5 with one estimator, but that group includes a residual column anchored on a SEARCH-fitted GP program. "
           f"Ultra-GP alone reaches {sysk['f_pure_ugp50']['elo']:.0f} with no ICL model and no trees. The ICL model without the GP anchor sits near TabICL v2, 180 Elo below the stack.", "body"))
S += fig("f02_strip", caption="Each dot is one scored arm of ours (hollow: fair). Vertical lines are five external models for reference. The two clusters in \"Neural columns\" are "
         "the H452 to H455 hybrids (right) and the r4 neural-plus-tabular arm (left); the two in \"ICL alone\" are one column family (left) and the CV-7 column set (right).")
S.append(P("Fair against in-corpus, matched pairs", "h2"))
S.append(P("Five pairs set a fair-closure arm beside its in-corpus twin. Two of them (H449) differ only in whether dev91 tables were in the ICL training corpus. "
           "The other three compare the H449 init with r4 checkpoints after 2 to 48 epochs of training at a 2% dev91 share, so they include that training too, and "
           "two of those three are one contrast under two stackers. The in-corpus arm is ahead by 1.6 to 8.3 Elo in every pair, against an interval of about 160 Elo: "
           "exposure to dev91 at that share, with up to 48 epochs of training on top, has been worth almost nothing.", "body"))
S += fig("f03_pairs", caption="Hollow: fair-closure arm. Filled: its in-corpus twin. The number at the left is the fair-closure arm's Elo; the bold number is the in-corpus advantage.")
S.append(PageBreak())

# ============================================================================= page 7: UGP + checkpoints
S.append(P("6. Ultra-GP generations", "h1"))
S.append(P(f"Ultra-GP evolves programs per context at inference using only that context's SEARCH rows. Alone it peaks at 50 generations "
           f"({sysk['f_pure_ugp50']['elo']:.0f}) and then loses about 20 Elo by 1,000 generations: the search keeps improving its own cross-validated fitness while "
           f"held-out accuracy falls. Inside the stack, 10 generations ties the stack without Ultra-GP ({T['ugp_base']['stack_no_ugp']:.1f}) and every higher count "
           f"is below it; the fair-closure stack ({T['ugp_base']['fair_no_ugp']:.1f} without Ultra-GP) loses 5 to 9 Elo with it.", "body"))
S += fig("f04_ugp", caption="Elo by generation count for three uses of Ultra-GP (H452 and H453 matrices). Line: the best admission variant at each count. Small dots: the other variants.")
S.append(P("7. The twelve H455 checkpoints", "h1"))
ck = [c for c in T["checkpoints"] if "stack" in c]
S.append(P(f"H455 scored twelve checkpoints and blends of the flywheel ICL models. The full stack moves by {max(c['stack'] for c in ck) - min(c['stack'] for c in ck):.1f} Elo "
           f"across them, the neural columns by {max(c['neural'] for c in ck) - min(c['neural'] for c in ck):.1f} and ICL alone by {max(c['icl'] for c in ck) - min(c['icl'] for c in ck):.1f}. "
           f"The checkpoint with the best ICL-alone score is not the one with the best stack. The ladder ranks each of these arms by its mean over the 12 "
           f"(stack {best['elo']:.1f}, neural columns {neural['elo']:.1f}, ICL alone {sysk['s_pure_cv7']['elo']:.1f}); the maxima are selection-biased.", "body"))
S += fig("f06_ckpt", caption="One row per scored checkpoint or blend (the scorers' labels). Each panel has its own narrow axis; the three levels are about 30 and 150 Elo apart.")
S.append(PageBreak())

# ============================================================================= page 8: r4
tr = T["traj"]
S.append(P("8. 48 hours of r4 training, epoch by epoch", "h1"))
S.append(P(f"H450 r4 trained two 104M-parameter ICL models for 48 epochs with dev91 in the corpus at a natural 2% share and is now finished. Over the init and {len(tr) - 1} scored epochs, the champion stack starts at "
           f"{tr[0]['champion']:.0f}, peaks at {max(t['champion'] for t in tr):.0f} and ends at {tr[-1]['champion']:.0f}. ICL alone moves from {tr[0]['icl_only']:.0f} to "
           f"{tr[-1]['icl_only']:.0f} (best {max(t['icl_only'] for t in tr):.0f}). The legacy guard stack, which uses the same ICL columns inside the GP and tree stack, "
           f"stays between {min(t['legacy'] for t in tr):.0f} and {max(t['legacy'] for t in tr):.0f}. All four series are in-corpus. The ladder ranks these "
           f"arms by their mean over the {sysk['s_r4_legacy']['n_epochs']} scored epochs after the init (legacy {sysk['s_r4_legacy']['elo']:.1f}, champion "
           f"{sysk['s_r4_champion']['elo']:.1f}, no GP {sysk['s_r4_nogp']['elo']:.1f}, ICL only {sysk['s_r4_icl']['elo']:.1f}); the maxima below are maxima over "
           f"{len(tr)} evaluations of one arm and are selection-biased. Its owner's verdict: the recipe is exhausted; 48 training hours moved the stack by at most 8 Elo.", "body"))
S += fig("f05_r4", caption="Four arms scored on the same epochs of H450 r4 (in-corpus), from the init to epoch 48, the final epoch. Source: r4_traj_final.tsv.")
rows = [["Arm", "Epoch 0", f"Max of {len(tr)} (biased)", "At epoch", f"Epoch {tr[-1]['epoch']}", f"Change 0 to {tr[-1]['epoch']}"]]
for key, name in (("legacy", "Legacy guard stack"), ("champion", "r4 champion stack"), ("no_gp", "Neural + tabular, no GP"), ("icl_only", "ICL only")):
    b = max(tr, key=lambda t: t[key])
    rows.append([name, f"{tr[0][key]:.1f}", f"{b[key]:.1f}", str(b["epoch"]), f"{tr[-1][key]:.1f}", f"{tr[-1][key] - tr[0][key]:+.1f}"])
S.append(table(rows, [2.2 * inch, 0.8 * inch, 1.3 * inch, 0.8 * inch, 0.9 * inch, 1.3 * inch], align_right=(1, 2, 3, 4, 5), font=7.6, pad=2.4))
m = T["r4_matched"]
S.append(Spacer(1, 6))
S.append(P(f"The run's own matched test (champion against its zero-dev91 control, epoch {tr[-1]['epoch']}): {m['elo_diff_champion_vs_zero_d91_ctrl_91']:+.1f} Elo, family mean gain "
           f"{m['pairs_91']['gain']:+.3f} u, families won, tied, lost {m['pairs_91']['wtl'][0]}, {m['pairs_91']['wtl'][1]}, {m['pairs_91']['wtl'][2]}. The same run's A/B of the "
           "max-entropy sampler against its control was negative at all nine scored epochs from 35 to 48 (-0.009 to -0.025 u on the ICL column; -0.022 at epoch 48, "
           "sign-flip p = 0.0001): the sampler cost own-column skill in the late, low-learning-rate phase. That is one run per arm.", "body"))
S.append(PageBreak())

# ============================================================================= page 9: head to head
S.append(P("9. Head to head", "h1"))
b_, c_, f_ = HD["best"], HD["fairc"], HD["fair"]
S.append(P("Elo on this board is decided by who wins each context, family by family, so the win counts are the most direct reading of where we stand. "
           f"Per-context scores exist for three systems of ours: the best arm at its best checkpoint ({b_['elo']:.0f}, in-corpus), the best fair-closure arm "
           f"({c_['elo']:.0f}, selected post hoc) and the pre-registered fair champion ({f_['elo']:.0f}).", "body"))
S.append(P(f"<b>Against Kumo Large all three win {H2H[KL]['best']['ctx_wtl'][0]} of 91 contexts</b> ({same_kumo} of them the same contexts), and Kumo Large wins every context "
           f"of {b_['kumo']['fam_swept_by_kumo']} families against each. The best arm wins {H2H['tabfm_ensemble.v1']['best']['ctx_wtl'][0]} contexts against TabFM ensemble and "
           f"{H2H['tabpfn_v3p5.v1']['best']['ctx_wtl'][0]} against TabPFN 3.5; from TabPFN v3 down it wins more than half. <b>We rank above TabPFN 3.5 on the ladder although "
           f"it beats us on {H2H['tabpfn_v3p5.v1']['best']['ctx_wtl'][2]} contexts head to head</b>: the board's rating counts games against all 33 entrants, 11 of them our own "
           "earlier systems (next page).", "body"))
S += fig("f09_wtl_ctx", caption="Contexts (of 91) where our system has the higher utility (blue, count at left) and where the external model does (red, count at right), "
         "for each of the 22 external entrants, ordered by the entrant's Elo. Per-context utilities of the best arm were released by the H455 scorer's owner on "
         "2026-11-15 and reproduce the published matrix exactly.")
rows = [["External model", "Elo", "Contexts W-L", "Games of 19", "Families W-L", "Contexts W-L", "Games of 19", "Contexts W-L", "Games of 19"]]
for r in T["h2h"]:
    cells = [r["name"].split(" (")[0] if len(r["name"]) > 34 else r["name"], f"{r['elo']:.0f}"]
    for k in ("best", "fairc", "fair"):
        cw, ct, cl = r[k]["ctx_wtl"]
        cells.append(f"{cw}-{cl}" + (f" ({ct} tied)" if ct else ""))
        cells.append(f"{r[k]['game']:.1f}")
        if k == "best":
            fw, ft, fl = r[k]["fam_wtl"]
            cells.append(f"{fw}-{fl}")
    rows.append(cells)
t = table([["", "", f"Best arm, best checkpoint ({b_['elo']:.0f})", "", "", f"Fair-closure arm ({c_['elo']:.0f})", "", f"Fair champion ({f_['elo']:.0f})", ""]] + rows,
          [1.72 * inch, 0.4 * inch, 0.76 * inch, 0.7 * inch, 0.74 * inch, 0.8 * inch, 0.7 * inch, 0.8 * inch, 0.68 * inch],
          align_right=(1, 2, 3, 4, 5, 6, 7, 8), font=6.2, pad=0.9)
t.setStyle(TableStyle([("SPAN", (2, 0), (4, 0)), ("SPAN", (5, 0), (6, 0)), ("SPAN", (7, 0), (8, 0)), ("ALIGN", (2, 0), (8, 0), "CENTER"),
                       ("FONTNAME", (0, 1), (-1, 1), "DVB"), ("TEXTCOLOR", (0, 1), (-1, 1), INK2), ("LINEBELOW", (0, 1), (-1, 1), 0.8, AXIS),
                       ("LINEBELOW", (0, 0), (-1, 0), 0, colors.white)]))
S.append(t)
S.append(P("Three counts of one comparison. <b>Contexts W-L</b>: contexts won and lost of 91. <b>Games of 19</b>: the ladder's own unit, the share of contexts won in each "
           "family, summed over the 19 families (9.5 is level). <b>Families W-L</b>: families where our mean utility is higher or lower, the count the scorers store with "
           "every arm and the one earlier revisions of this report quoted (1 of 19 against Kumo Large). The fair-closure arm is H453 fair_oos_0pct_d91_no_ugp.", "cap"))
S.append(PageBreak())

# ============================================================================= page 10: external models only
lb, lc, lf = LG["best"], LG["fairc"], LG["fair"]
ltab = {r["id"]: r["games"] for r in lb["table"]}
dc = T["tabpfn_decomp"]
S.append(P("Against external models only", "h1"))
rtab = {r["id"]: r["games"] for r in LG["recall"]["table"]}
S.append(P(f"<b>Against the 22 external models alone, and without recall columns, we are seventh of {lb['n']}, not fifth.</b> The board's 33 entrants include 11 of our own earlier systems. Leave those "
           f"out, and each of the {lb['n']} remaining entrants still plays 19 family games against every other, so the rating order is simply the order of games won: Kumo Large "
           f"{ltab[KL]:.0f} of {lb['of']}, TabFM ensemble {ltab['tabfm_ensemble.v1']:.0f}, Kumo Small {ltab['kumo_tabular_small_e8.v1']:.0f}, TabFM {ltab['tabfm.v1']:.0f}, TabPFN 3.5 "
           f"{ltab['tabpfn_v3p5.v1']:.0f}, TabPFN 3.5 with one estimator {ltab['tabpfn_v3p5_e1.v1']:.0f}, then our best arm at {lb['games']:.0f}. The fair-closure arm "
           f"({lc['games']:.0f}) and the pre-registered fair champion ({lf['games']:.0f}) take the same seventh place, ahead of TabPFN v3 ({ltab['tabpfn_v3.v1']:.0f}). To pass "
           f"TabPFN 3.5 with one estimator the best arm needs {ltab['tabpfn_v3p5_e1.v1'] - lb['games']:.0f} more games from the other 21 models, or about "
           f"{(ltab['tabpfn_v3p5_e1.v1'] - lb['games']) / 2:.0f} taken from it head to head (those count twice); to pass TabPFN 3.5, {ltab['tabpfn_v3p5.v1'] - lb['games']:.0f} "
           f"or {(ltab['tabpfn_v3p5.v1'] - lb['games']) / 2:.0f}. Kumo Large is {ltab[KL] - lb['games']:.0f} games ahead and cannot be passed on head-to-head games alone. "
           f"<b>With the family recall specialists (in-corpus recall: audit rows trained on, with labels) the stack takes {LG['recall']['games']:.0f} of {LG['recall']['of']} games "
           f"and is first of {LG['recall']['n']}</b>; Kumo Large has {rtab[KL]:.0f} in that league.", "body"))
S += fig("f08_league", caption=f"Family-games won against the 22 external models, of {lb['of']} (22 opponents, 19 dataset families each; a game is the share of a family's contexts "
         "won). Grey: external models, in the league that contains our best arm. Blue: our headline systems, each rated alone against the 22; filled is in-corpus, "
         "outlined is fair closure, hatched is in-corpus recall (the H458 look, whose specialists were trained on these audit rows). The external bars move by less than "
         f"two games between the first three leagues; in the league with the recall arm each external model has up to {max(ltab[k] - rtab[k] for k in ltab if k in rtab and k != 'OURS'):.0f} "
         f"games fewer (Kumo Large {rtab[KL]:.0f}).")
S.append(P("Where fifth place on the board comes from", "h2"))
S.append(P(f"On the board our best arm, at its best checkpoint, sits {T['top50'][0]['gap_tabpfn35']:.0f} Elo above TabPFN 3.5, yet TabPFN 3.5 wins "
           f"{dc[3]['tabpfn35']:.1f} of the 19 games between the two. With the opponents fixed, a Bradley-Terry rating depends only on the total of games won, and over all 33 "
           f"opponents the best arm collects {dc[4]['ours']:.1f} games to TabPFN 3.5's {dc[4]['tabpfn35']:.1f}. The table shows where that margin of {dc[4]['diff']:+.1f} comes from: "
           "we gain it on our own earlier systems and on the classical and LLM entrants, and lose most of it again on the other foundation models.", "body"))
rows = [["Opponents", "Games played", "Our best arm wins", "TabPFN 3.5 wins", "Difference"]]
for r in dc:
    rows.append([r["opponents"], str(r["of"]), f"{r['ours']:.1f}", f"{r['tabpfn35']:.1f}", f"{r['diff']:+.1f}"])
S.append(table(rows, [3.3 * inch, 0.95 * inch, 1.15 * inch, 1.05 * inch, 0.85 * inch], align_right=(1, 2, 3, 4), font=7.4, pad=2.2, bold_rows=(len(rows) - 1,)))
S.append(P("Games are family-games as above. The best arm is taken at its best checkpoint, the only one with per-context scores. This decomposition and the external-only "
           "order were first computed by Track S on 2026-11-15 and recomputed here.", "cap"))
S.append(PageBreak())

# ============================================================================= page 10: families
S.append(P("10. Dataset families: who is best where", "h1"))
top3 = [f for f, r in HD["best"]["fam_rank"].items() if r <= 3]
S.append(P(f"Kumo Large has the best family mean utility on 10 of 19 families and is in the top three on 15. The other nine families are won by eight different models, "
           f"including Ridge on one. Without recall columns no system of ours is best on any family: the best arm would place in the top three on {len(top3)} ({', '.join(sorted(top3, key=lambda f: HD['best']['fam_rank'][f]))}) "
           f"and is above every field entrant on {HD['best']['ctx_above_field']} of 91 contexts. With the family recall specialists (in-corpus recall: audit rows trained on, "
           f"with labels) the stack has the best family mean in {L2['n_fam_first']} of 19 families and is above every entrant on {L2['ctx_above_field']} contexts (section 13); "
           "the chart and the table below are without them.", "body"))
S += fig("f07_heat", caption="Rank of each model's family mean utility among the 33 field entrants (dark = better; ranks 1 to 3 are printed). Our two newer arms (bold, rows 5 "
         "and 6) are shown at the rank they would take if added to the field. Columns are the 19 dataset families.")
rows = [["Family", "Ctx", "Feat.", "Best field entrant", "Best u", "Kumo L", "Best arm", "Rank", "Ctx won", "Fair ch.", "Rank", "Gap"]]
for r in sorted(T["family_table"], key=lambda r: r["best_arm_gap_to_best"]):
    rows.append([r["family"], str(r["n_ctx"]), str(r["width"]), r["best"].split(" (")[0], f"{r['best_u']:.3f}", f"{r['kumo_l_u']:.3f}",
                 f"{r['best_arm_u']:.3f}", str(r["best_arm_rank"]), f"{r['best_arm_ctx_vs_kumo']} of {r['n_ctx']}", f"{r['fair_u']:.3f}", str(r["fair_rank"]),
                 f"{r['best_arm_gap_to_best']:+.3f}"])
t = table(rows, [0.72 * inch, 0.3 * inch, 0.38 * inch, 1.6 * inch, 0.52 * inch, 0.55 * inch, 0.6 * inch, 0.4 * inch, 0.58 * inch, 0.56 * inch, 0.4 * inch, 0.69 * inch],
          align_right=(1, 2, 4, 5, 6, 7, 8, 9, 10, 11), font=6.5, pad=1.2)
t.setStyle(TableStyle([("FONTNAME", (0, 1), (0, -1), "DVM")]))
S.append(t)
S.append(P("All utilities are family means. \"Best arm\" is our best arm at its best checkpoint (in-corpus) and the \"Rank\" beside it the place it would take among "
           "the field entrants on that family. \"Ctx won\" is the number of the family's contexts on which the best arm beats Kumo Large. \"Fair ch.\" is the "
           "pre-registered fair champion (H448) with its rank among the 33. \"Gap\" is the best arm minus the best external model; rows are ordered by it, "
           "largest deficit first.", "cap"))
S.append(PageBreak())

# ============================================================================= page 11: contexts
kb, kf = HD["best"]["kumo"], HD["fair"]["kumo"]
ft_ = {r["family"]: r for r in T["family_table"]}
ratio = [__import__("math").exp(ft_[f]["kumo_l_u"] - ft_[f]["best_arm_u"]) for f in kb["top5"][:2]]
S.append(P("11. Context by context against Kumo Large", "h1"))
S.append(P(f"Our best arm is better than Kumo Large on {kb['ctx_wins']} of 91 contexts and on every context of one family ({', '.join(kb['fam_won_all'])}); Kumo Large "
           f"wins every context of {kb['fam_swept_by_kumo']} families. The deficit is concentrated: the five families with the largest summed deficit "
           f"({', '.join(kb['top5'])}) hold {100 * kb['top5_share']:.0f}% of it, and the first two are near-noiseless tables where Kumo Large's error is "
           f"{min(ratio):.1f} to {max(ratio):.1f} times smaller than ours. On {kb['n_close']} of the 19 families the two models are within 0.15 u of each other on every "
           f"context, yet Kumo Large is ahead on every context of {kb['n_close_swept']} of those, and the ladder counts each as a full loss. The fair champion's picture is "
           f"almost the same ({kf['ctx_wins']} contexts won, {100 * kf['top5_share']:.0f}% of its deficit in the same five families): the {HD['best']['elo'] - HD['fair']['elo']:.0f} "
           "Elo between the two systems came from the rest of the field, not from Kumo Large.", "body"))
S += fig("f10_ctx", caption="Utility of our system minus utility of Kumo Large, one bar per context, grouped by dataset family. Negative: Kumo Large is better. "
         "Top: our best arm at its best checkpoint. Bottom: the pre-registered fair champion.")
S.append(P("Why the size of a loss does not matter on this ladder, and its sign does", "h2"))
S.append(P("The ladder converts every context into a win, a tie or a loss and averages those within a family; the margin is discarded. Closing a 0.05 u gap on a family "
           "where we lose all five contexts is worth as much as closing a 1.5 u gap. That is why 18 narrow family losses cost as much as 18 wide ones, and why the route "
           "to first place runs through winning whole families, not through raising mean utility.", "body"))
S.append(PageBreak())

# ============================================================================= page 12: what #1 takes + H458
S.append(P("12. What first place takes", "h1"))
S.append(P("A simulation with the ladder's own code on the stored scores (H458 plan, 2026-11-14). A \"win\" here means beating all 33 field entrants on a context, "
           f"because the bar on each context is the best of the field, not Kumo Large alone. Without recall columns the best arm clears that bar on {HD['best']['ctx_above_field']} of 91 "
           f"contexts; with the family recall specialists (in-corpus recall) the stack clears it on {L2['ctx_above_field']}.", "body"))
S.append(table([
    ["Scenario", "Our Elo", "Kumo L Elo", "Our rank"],
    ["Without recall columns (best arm, mean of 12 checkpoints)", "1862", "2078", "5"],
    ["Win the 12 cheapest families, but only against Kumo Large", "2009", "2053", "2"],
    ["Win the 12 cheapest families against the whole field", "2068", "2058", "1, barely"],
    ["Win the 10 small-source families against the whole field", "2010", "2054", "2"],
    ["Win 16 families (the 12 cheapest plus 469b, ce33, 94f2, 866f) against the whole field", "2379", "2071", "1"],
    ["Win a random 50%, 70%, 85% of contexts against the whole field", "2042, 2155, 2268", "about 2062", "rank 1 in 1, 5 and 5 of 5 draws"],
], [3.75 * inch, 1.05 * inch, 0.8 * inch, 1.7 * inch], align_right=(1, 2), font=7.2, pad=2.2))
S.append(Spacer(1, 5))
S.append(P("Two bounds computed by Track S on the stored columns put the gap in perspective. Choosing the best existing column per context after the fact "
           "reaches 1998, so better selection alone cannot reach Kumo Large with today's columns. A uniform lift of +0.05 utility on every context would put the "
           "legacy stack at 2002, and +0.10 at 2082, above Kumo Large's 2070: the gap is about a tenth of a unit of utility, on every context. "
           f"<b>What the look of 2026-11-15 measured (in-corpus recall):</b> {L2['n_swept']} families won against the whole field on every context and partial wins in "
           f"{sum(1 for r in L2['families'] if 0 < r['above_field'] < r['n'])} more gave {L2['elo']:.0f} against Kumo Large at {KL_FIT:.0f}, between the third and fifth rows of the table.", "body"))
S += fig("f17_ctx_recall", caption="Utility minus that of Kumo Large, one bar per context, grouped by dataset family. Top: the stack without the recall columns. Bottom: the "
         "same stack with one recall specialist per family (in-corpus recall: these audit rows were trained on, with labels). Kumo Large keeps every context of the two "
         "nearly noiseless families 4286560c and 280b1a13.")
S.append(PageBreak())
S.append(P("13. The H458 family specialists: the full ladder, as in-corpus recall", "h1"))
w0, w1, w2 = L2["pair"]["wtl"]
S.append(P(f"<b>In-corpus recall: {L2['elo']:.0f} Elo and first of 34; the gate set before the look was not met.</b> Wave H458 fine-tunes one copy of the 23.7M-parameter ICL "
           "model per dev91 family on that family's pooled rows, audit rows and their labels included. At inference a specialist reads only a context's own SEARCH rows. "
           "The owner's rule allows this, and the result carries the label in-corpus recall wherever it appears, because every audit row it is scored on sat in its "
           f"training pool. One pre-registered look on all 91 contexts, the second of the four the campaign may spend, gave the legacy stack with the specialists' columns "
           f"{L2['elo']:.1f} Elo (95% interval {L2['ci'][0]:.0f} to {L2['ci'][1]:.0f}), rank 1 of 34 and {L2['gap_kumo']:.0f} above Kumo Large in the same fit (in-corpus recall). "
           f"The same stack without those columns is {L2['elo_nr']:.1f}, rank {L2['rank_nr']}. The gain is {L2['pair']['gain']:+.3f} utility on the family mean, with families "
           f"won, tied, lost {w0}, {w1}, {w2} (sign-flip p = {L2['pair']['signflip_p']:.1e}). The stack put {L2['w_recall_mean'] * 100:.0f}% of its weight on the specialist "
           f"columns and is below the stack without them on {L2['ctx_below_base']} contexts.", "body"))
S += fig("f14_look2", caption="In-corpus recall. Family mean utility on AUDIT rows of the stack without the recall columns (hollow), of the same stack with one recall "
         "specialist per family (filled; the specialists were trained on these rows, with labels) and of the best of the 33 field entrants (black tick). Beside each "
         "family: contexts on which the stack with specialists is above every entrant. * marks the three families the plan did not count on. Look "
         "h458d.RECALL_CANON_E2000_ALL91, pre-registered 03:14 UTC and run 03:27 to 03:33 UTC on 2026-11-15.")
rows = [["Family", "Above the field", "With specialists", "Stack", "Field best", "Rank", "Weight", "Best entrant on the contexts not won (contexts)"]]
for r in sorted(L2["families"], key=lambda r: -(r["u_arm"] - r["field_max"])):
    rows.append([r["family"] + ("" if r["target"] else " *"), f"{r['above_field']} of {r['n']}", f"{r['u_arm']:.2f}", f"{r['u_base']:.2f}", f"{r['field_max']:.2f}",
                 str(r["rank"]), f"{r['w_recall'] * 100:.0f}%", P(", ".join(f"{short(h['name'])} ({h['n']})" for h in r["held_by"]), "cell")])
t = table(rows, [0.85 * inch, 0.8 * inch, 0.9 * inch, 0.5 * inch, 0.65 * inch, 0.45 * inch, 0.55 * inch, 2.6 * inch], align_right=(1, 2, 3, 4, 5, 6), font=6.7, pad=1.5)
t.setStyle(TableStyle([("FONTNAME", (0, 1), (0, -1), "DVM")]))
S.append(t)
S.append(P("In-corpus recall. Utilities are family means over audit rows. Above the field: contexts on which the stack with specialists beats every one of the 33 "
           "entrants. Field best: the mean of the per-context best of the 33. Rank: the stack with specialists among the 33 entrants by family mean utility. Weight: the "
           "stack's weight on the specialist columns. * not a target family.", "cap"))
miss = [r for r in L2["families"] if r["target"] and not r["swept"]]
S.append(P(f"<b>Gate D-4 is not met.</b> It was written before the look: the stack with specialists above every one of the 33 entrants on every context of at least "
           f"{L2['gate_needed']} of the {L2['n_target']} target families, and an Elo above Kumo Large. The Elo clause holds. {L2['n_swept_target']} of the {L2['n_target']} target "
           "families are swept. The others: "
           + "; ".join(f"{r['family']} ({r['above_field']} of {r['n']}; best on the rest: " + ", ".join(f"{short(h['name'])} on {h['n']}" for h in r["held_by"]) + ")" for r in miss)
           + ". On 769243ee the stack gives the specialist columns no weight, and on 9f67f144 about a sixth of the rows share their features with rows of a different target, "
           "which caps what recall can do. The expectation written into the pre-registration was 10 to 13 sweeps, the gate more likely missed than met, and rank 1 or 2.", "body"))
S += fig("f15_look2_wtl", caption="Contexts (of 91) won and lost against each of the 22 external entrants: the stack without the recall columns, and with one recall "
         "specialist per family (in-corpus recall: these audit rows were trained on, with labels).")
S.append(P(f"The first look, on 20 contexts of four pilot families (2026-11-14), had the stack with specialists above the whole field on {PL['wins']} of {PL['n']}; the full "
           "look contains those contexts again.", "body"))
S.append(P("What transfers to rows the specialists never saw", "h2"))
md = PRB["modes"]
S.append(P(f"<b>On fresh rows the specialists beat their own starting checkpoint by {PRB['du']:+.2f} utility.</b> Track S built a separate board from rows of the same source "
           f"tables that are in no bank and no training pool ({PRB['n_families']} families, {PRB['n_contexts']} contexts, about 20,000 rows; no fresh row shares its full feature "
           "vector with a pool row). Three arms were named in that board's ledger before any scoring and scored once. The canonical-slot specialists reach "
           f"{PRB['u_canon']:.2f} family-mean utility against {PRB['u_init']:.2f} for the checkpoint they were fine-tuned from: {PRB['ctx_wtl'][0]} of {PRB['n_contexts']} "
           f"contexts and {PRB['fam_wtl'][0]} of {PRB['n_families']} families, sign-flip p = {PRB['p']:.3f} two-sided on {PRB['n_families']} families. By target mode, "
           f"specialist against init: ridge-residual {md['rr']['canon']:.2f} against {md['rr']['init']:.2f}, Yeo-Johnson {md['yj']['canon']:.2f} against {md['yj']['init']:.2f}, "
           f"rank-Gauss {md['rg']['canon']:.2f} against {md['rg']['init']:.2f}. On 60b608db the random-slot specialist gains {PRB['b60']['rand']:+.2f} and the canonical one "
           f"{PRB['b60']['canon']:+.2f} over an init at {PRB['b60']['init_u']:.2f}.", "body"))
S += fig("f16_freshprobe", caption=f"Fresh-row board, {PRB['n_families']} families: utility of the family specialist minus that of its own pre-fine-tune checkpoint, family mean over "
         "the family's contexts. The largest gains are on the two nearly noiseless tables; three families are lower. Arms h458d.FRESH_*, pre-registered 2026-11-15 00:12 UTC, "
         "scored once at 03:12 UTC.")
sk = H["skill"]
g = lambda arm, col: ", ".join(f"{r[col] - r['base']:+.3f}" for r in sk[arm])  # noqa: E731
S.append(P("<b>The prediction registered before scoring was the opposite, and it is recorded as refuted.</b> It came from a cross-fitted control on family 60b608db, in which "
           f"each row is scored by a model trained without that row's feature stratum: random slots {g('rand', 'e1000')} utility at 1,000 exposures and canonical slots "
           f"{g('canon', 'e2000')} at 2,000. That control removes a whole stratum of the table; the fresh-row board removes rows at random within it, and there the specialists "
           "are better. What the result supports: fine-tuning on a family's pooled rows, two to six times the rows of one context, teaches the model something about the "
           "table and not only its audit labels. What it does not support: any claim about a table without a specialist, or a fair comparison with an external model. "
           "Limits: the pools hold dev91 audit labels; a specialist is scored with one native-order view and its init with the evaluation's four random views; rows that "
           f"match a pool row on all columns but one are not excluded; and {PRB['n_families']} families give p = {PRB['p']:.3f}, which is not a settled effect.", "body"))
hf = H["families"]
S.append(P(f"How the specialists learn, on SEARCH rows only: after 2,000 query exposures per row the canonical-slot specialists reach {hf[0]['canon'][-1]['u']:.2f}, "
           f"{hf[1]['canon'][-1]['u']:.2f} and {hf[2]['canon'][-1]['u']:.2f} out of fold on three pilot families, from {hf[0]['init']['canon']:.2f}, "
           f"{hf[1]['init']['canon']:.2f} and {hf[2]['init']['canon']:.2f} before fine-tuning.", "body"))
S += fig("f12_h458", width=W * 0.86, caption="Out-of-fold utility on SEARCH rows by training exposure, for specialists with canonical and with random feature slots. Grey line: the best "
         "field entrant's family mean utility on AUDIT rows, for scale.")
S.append(PageBreak())

# ============================================================================= page 13: tonight + method
S.append(P("14. Other results of the two nights", "h1"))
S += bullets([
    "<b>Staged residual GP (S9, the owner's experiment of 2026-11-15), in progress.</b> A chain of GP runs, each fitted to the residual of the running sum, is being tested "
    "against single runs of the same and of twice the budget on a separate board of 80 contexts (P201B). The 40-generation tier is complete and handed to Track S "
    "for its one stack-level scoring; the 160-generation tier was stopped once by its own host-load rule and restarted at 09:00 UTC on 2026-11-15. "
    "A descriptive pilot on 10 contexts read on fresh rows: the plain four-stage chain beat a single run on 7 of 10 contexts (two of them by under 0.002 utility; "
    "mean +0.17, p = 0.50), its curve was flat after the second stage, a single run with one and a half times the generations was ahead of every chain once the "
    "column was stacked, and a least-squares blend of one run's own three best programs beat the plain single run on 8 of 10. One extreme row can carry a "
    "context's whole error: 22 of the 80 P201B contexts have a target more than 8 standard deviations from the mean.",
    f"<b>Stage 2 (H455) finished at 22:59 UTC.</b> Six checkpoint matrices: best stack {sysk['s_stack_ugp1000']['elo_best']:.1f} (L-x1 v2 valley 001), best neural-column arm "
    f"{neural['elo_best']:.1f} (L and M blended 50/50). The highest single scoring is still the Stage 1 M-x2 valley average at {best['elo_best']:.1f}.",
    f"<b>H450 r4 finished at epoch {tr[-1]['epoch']}:</b> champion {tr[-1]['champion']:.1f}, ICL only {tr[-1]['icl_only']:.1f}, legacy guard stack {tr[-1]['legacy']:.1f}; no new best. "
    "Both trainers are final and their GPUs are free.",
    "<b>Track S (stack and selection):</b> 12 stack and guard variants were tested against the legacy baseline on a separate non-dev91 board (P201B); none passed its pre-declared gate, so no dev91 "
    "look was spent and the legacy stack stays the base. An oracle that picks the best existing column per context after the fact reaches 1997.6 on dev91, which bounds "
    "what better selection alone can give with today's columns.",
    "<b>Track G (GP flywheel):</b> on 11,912 teaching episodes, 6.5% of GP teachers that won on cross-validation lost to a ridge on held-out rows; a filtered teacher set "
    "with no such losers and a held-out-trained surrogate are now feeding the harvester. No dev91 score yet.",
    "<b>New training run X3-B</b> started on a third GPU at 23:11 UTC (evaluation-aligned episodes, random feature slots). No score yet.",
], "bullet_s")
S.append(P("Method", "h1"))
S += bullets([
    "<b>Board.</b> dev91-audit-v1: 91 contexts in 19 dataset families. A model fits on a context's SEARCH rows (566 to 1,183) and predicts its AUDIT rows.",
    "<b>Utility.</b> u = -ln(RMSE on audit rows / RMSE of predicting the SEARCH mean). Zero is the trivial predictor; 1.0 is an RMSE 2.7 times smaller.",
    "<b>Elo.</b> A Bradley-Terry fit in which each family is one game between each pair of entrants, scored as the mean over the family's contexts of win, tie or loss "
    "on the sign of the utility difference. The anchor is the H257 replica at 1500. Intervals are the scorers' bootstrap intervals (2,000 to 10,000 resamples).",
    "<b>One scale.</b> External models and earlier systems keep their rating from the frozen 33-entrant fit. Each newer arm of ours is inserted alone into that field "
    "(34 entrants), which is how every scorer in H449 to H455 already worked. A joint fit of all arms was rejected because 50 near-identical stacks would distort the "
    "external ratings.",
    "<b>In-corpus recall.</b> The H458 look is a single pre-registered scoring of one arm (ledger line 3 before, line 4 after; the second of four looks the campaign "
    "may spend). Its Elo comes from inserting that arm alone into the frozen field, like every other arm. Games against the 22 external models are computed from "
    "its per-context utilities with the ladder's own tie rule.",
    "<b>Repeated scorings.</b> A system scored more than once (34 r4 epochs, 12 H455 checkpoints, repeated matrices) is ranked by its mean, the value the "
    "owners of those results asked for. The maximum over repeated noisy scorings is selection-biased; it appears only in sections 3, 7 and 8.",
    "<b>Independent check.</b> Track S re-rated 253 arms from per-context utilities, all equal to the reported Elo within 4.5e-13: 241 from stored scores, the eight "
    "H455 arms whose per-context utilities their scorer's owner released on 2026-11-15 (the 1865.36 maximum included), and the four epoch-48 arms. The field-only fit "
    "reproduces the stored ladder exactly. The head-to-head counts of section 9 were derived independently by three agents.",
], "bullet_s")
S.append(P("Caveats", "h2"))
S += bullets([
    "The H458 recall number is not evidence of skill on unseen tables. Every audit row it is scored on was a training example with its label, by the owner's rule "
    "for training. The only read of the same specialists on rows they never saw is the fresh-row board: ten families, p = 0.055.",
    "The intervals are wide: 19 families give about 160 Elo either side. Differences of under 50 Elo between two arms of ours are not established by this board alone.",
    "Only the H448 fair champion was pre-registered; every other fair-closure row was selected after dev91 results were visible and is optimistic by an unknown amount.",
    "Fair closure means no dev91 table in training. Two wider exposure questions are open and apply to those rows too (the H449 init included): 281 tables and 258 views from a sealed partition, and one name-matched view, were admitted to the "
    "training bank of the H449 and H450 lineage, and the stored GP pools were built by workers that had audit labels in memory (a permutation test of that is pending).",
    "dev91 is a reuse board: the H257 system was tuned on it and every arm here was selected with it in view. It is not a clean confirmation set.",
    "A third of the board's entrants (11 of 33) are our own earlier systems, which our newer arms beat easily. That lifts our board rank above models that beat us head "
    "to head; the external-only league in section 9 removes it.",
    "The Ultra-GP columns of H453 and H455 were evolved with the engine's default surrogate prior, which was fitted with 289 winners of dev91 random splits (SEARCH "
    "and AUDIT rows mixed). Fair-closure rows that contain Ultra-GP columns (Ultra-GP alone; fair stack + Ultra-GP) are therefore fair only as far as the ICL corpus "
    "goes. The 1853 fair-closure arm has no Ultra-GP column and is not affected. (Found in review on 2026-11-15; a dev91-free prior exists and new runs use it.)",
    "Our systems have no wall time on this board, so the cost chart cannot place them.",
], "bullet_s")
S.append(PageBreak())
S.append(P("15. After revision 9: Stage 3, seed averaging, and what our own ICL model adds", "h1"))
S.append(P(f"<b>Nothing in this section changes a rank.</b> The pre-registered fair champion stays at {fld['h448.APEX_FAIR_CHAMPION']['elo']:.0f}, seventh on the board and "
           f"{LG['fair']['rank']}th of {LG['fair']['n']} against external models only. The stack arms stay in-corpus. What is new is evidence on which levers are worth "
           "building next, and three ideas that were tested and closed.", "lead"))
S.append(P("Stage 3 checkpoints (H456), in-corpus", "h2"))
rows = [["ICL columns from", "Stack", "95% interval", "Board", "Externals only\n(games)", "Contexts won\nfrom Kumo L", "Stack +\nUltra-GP", "Neural\ncolumns", "ICL\nalone"]]
for r in R10["stage3"]:
    rows.append([r["label"], f"{r['stack']:.1f}", f"{r['stack_ci'][0]:.0f} to {r['stack_ci'][1]:.0f}", f"{r['stack_rank']} of 34", f"{r['league']['rank']} of {r['league']['n']} "
                 f"({r['league']['games']:.0f} of {r['league']['of']})", f"{r['kumo']['ctx_wtl'][0]} of 91", f"{r['stack_ugp']:.1f}", f"{r['neural']:.1f}", f"{r['icl']:.1f}"])
S.append(table(rows, [1.75 * inch, 0.5 * inch, 0.8 * inch, 0.5 * inch, 1.05 * inch, 0.95 * inch, 0.75 * inch, 0.65 * inch, 0.5 * inch], align_right=(1, 2, 3, 4, 5, 6, 7, 8), font=6.7, pad=1.6))
S.append(P(f"In-corpus: the ICL models' training bank held dev91 tables. Each arm is inserted alone into the frozen field of 33. The highest Stage 3 scoring is "
           f"{R10['stage3_best']['elo']:.1f} (interval {R10['stage3_best']['ci'][0]:.0f} to {R10['stage3_best']['ci'][1]:.0f}), below the Stage 1 scoring of {best['elo_best']:.1f}; "
           "the five Stage 3 stacks lie within 2 Elo of each other and within 4 of the Stage 1 scorings. The ICL model alone is about 175 Elo below the stack that contains it.", "cap"))
S.append(P("Averaging two seed runs (priced by the APEX agent, H462)", "h2"))
rows = [["Stack", "Label", "One run", "Two runs\naveraged", "Gain", "Board\nrank", "Families better,\nsame, worse"]]
for r in R10["seedavg"]:
    rows.append([P(r["label"], "cell"), P(r["tag"], "cell"), f"{r['before']:.1f}", f"{r['after']:.1f}", f"{r['gain']:+.1f}", f"{r['rank_after']} of 34",
                 f"{r['fam_wtl'][0]}, {r['fam_wtl'][1]}, {r['fam_wtl'][2]}"])
S.append(table(rows, [2.1 * inch, 1.9 * inch, 0.55 * inch, 0.9 * inch, 0.45 * inch, 0.65 * inch, 0.95 * inch], align_right=(2, 3, 4, 5, 6), font=6.7, pad=1.6, bold_rows=(1,)))
S.append(P("Elo of each arm inserted alone into the frozen field, before and after averaging the predictions of the two existing seed runs. No new training. The two rows "
           "without the r4 epoch-16 column were first posted as free of dev91; their ICL model was pre-trained with a 1.8% share of dev91 rows, test rows included, so they "
           "are in-corpus. On tables where one seed finds an exact formula and the other does not, a plain average loses the exact one; the product needs a guard there.", "cap"))
S.append(P("Ten contexts, fresh rows: what our ICL model adds", "h2"))
pv = P10["vs_init"]
S.append(P(f"<b>Fine-tuning on a table's own SEARCH rows works; training on other tables does not; and a small tree is as good as the frozen model with help.</b> One context "
           "from each of ten families, scored on rows that are in no bank and no training pool. Every arm was named in the probe ledger with its pass rule before its "
           f"columns existed, and scored once. The frozen starting model is at {P10['init_u']:.2f} mean utility. Fine-tuned on the table's own SEARCH rows with random "
           f"feature slots it reaches {P10['own_u']:.2f}: {pv[0]['w']} of {P10['n']} contexts, {pv[0]['du']:+.2f}, sign-flip p = {pv[0]['p']:.3f}. A generalist trained for the "
           f"same number of steps on the other nine tables does not move ({pv[6]['du']:+.2f}, {pv[6]['w']} of {P10['n']}), and the own-table model beats it on "
           f"{P10['own_vs_lofm']['w']} of {P10['n']} ({P10['own_vs_lofm']['du']:+.2f}). So the gain is adaptation to the table, not a better match between training and "
           "evaluation. The prediction registered beforehand was the opposite.", "body"))
S += fig("f17_rev10_probe", caption=f"Fresh-row board, ten contexts (one per family): each arm's utility minus the frozen starting model's ({P10['init_u']:.2f}), mean over "
         "contexts, with contexts won and lost. Scored once each between 07:08 and 07:58 UTC on 2026-11-15. Ten contexts give a direction, not a settled effect.")
S += bullets([
    f"<b>The frozen model uses hints and adds nothing to them.</b> Given a ridge, a nearest-neighbour and a boosted-tree prediction as three extra input columns, it beats "
    f"itself without them on {pv[4]['w']} of {P10['n']} ({pv[4]['du']:+.2f}) and beats the best of its own three hints on {P10['hints_above_best']} of {P10['n']}. "
    "The first count is an upper bound: a fresh row's hint was the mean of five fold models and a context row's hint one fold model (review of 2026-11-15, 14:11 UTC).",
    f"<b>The small boosted tree alone is at {P10['tree_u']:.2f}</b> and beats the frozen model on {pv[2]['w']} of {P10['n']}. The fine-tuned "
    f"model is above the tree in the mean ({P10['own_u']:.2f}) and on 4 of the 10 contexts; that contrast was not registered and is a description.",
    f"<b>With canonical feature slots, more fine-tuning made it worse:</b> {pv[1]['du']:+.2f} at 500 exposures per row, {pv[3]['du']:+.2f} at 1,000, {pv[5]['du']:+.2f} at 2,000 "
    f"({pv[5]['w']} of {P10['n']}; this recipe failed its rule). A model fine-tuned on a different table is below the generalist on {P10['foreign_vs_lofo']['l']} of {P10['n']}.",
    "<b>Not yet tested:</b> whether the fine-tuned column still adds anything once it sits in the stack beside the tree and GP columns. That is the next registered read.",
], "bullet_s")
S.append(P("What a gain would be worth: replays of the frozen board", "h2"))
rows = [["If the fair champion were changed like this", "Elo", "Board rank", "Games of 418 against\nthe 22 externals"]]
for r in R10["price"]:
    rows.append([P(r["label"], "cell"), f"{r['elo']:.0f}", f"{r['rank']} of 34", f"{r['games']:.0f}"])
S.append(table(rows, [4.3 * inch, 0.6 * inch, 0.8 * inch, 1.75 * inch], align_right=(1, 2, 3), font=6.7, pad=1.5, bold_rows=(1,)))
S.append(P("Replays from the stored per-context utilities of the frozen board: no model was run, no label read, and none of these is a result. They are upper bounds that "
           "assume a perfect choice per context. Kumo Large is at about 2,060 to 2,080 in these fits. The fair champion is below one of five classic models on "
           f"{R10['floor_below']} of {R10['floor_n']} contexts. A rule that needs no choice, fielding CatBoost wherever the target has a heavy tail, replays at 138 Elo "
           "below the champion, so the floor is only worth having with a selector that is right.", "cap"))
S.append(P("Closed on 2026-11-15", "h2"))
S += bullets([
    "<b>A search for exact low-degree laws on the precision tables</b> (APEX agent, H462): accepted on 0 of 91 dev91 contexts under its own rule (+0.5 utility on five of "
    "five folds, SEARCH rows only). Its ceiling in the replay above was 55 Elo.",
    "<b>An exact per-table solve on the frozen ICL model's features</b> (APEX agent, H462): 0.06 utility below the model's own output on 121 design contexts.",
    "<b>A generalist retrained on evaluation-matched episodes as the next step</b>: its rule failed on the ten contexts, at equal exposure and at matched steps.",
], "bullet_s")
S.append(PageBreak())
S.append(P("16. Later on 2026-11-15: what is left standing", "h1"))
SR = R11["series"]
S.append(P(f"<b>One lever has measured lift: averaging our own runs. Everything else tested in two days was null or absorbed.</b> With 0% dev91 in training, the fair entrant "
           f"went from {SR[0]['elo']:.1f} to {SR[-1]['elo']:.1f} Elo, seventh to sixth of 33 on the board. Four agents now agree on the next wave: more members until the "
           "gain falls below seed noise, an integrity check and a forecast written before the look, then one dev91 look on one locked entrant, only if it clears the stop "
           "rule and the forecast's lower bound is above zero; otherwise no look. Its expected gain is about 5 Elo. First place under the ours-only rule would need about +0.15 utility on every context, and nothing we hold moves a column that far.", "lead"))
S.append(P("The fair series from averaging (APEX agent, wave H462)", "h2"))
rows = [["Entrant, 0% dev91 in training", "Elo", "Board rank", "Status"]]
for r in SR:
    rows.append([P(r["label"], "cell"), f"{r['elo']:.1f}", f"{r['rank']} of 33", P(r["note"], "cell")])
S.append(table(rows, [3.0 * inch, 0.6 * inch, 0.8 * inch, 3.05 * inch], align_right=(1, 2), font=6.7, pad=1.6, bold_rows=(1, len(rows) - 1)))
S.append(P(f"A dev91 development read: seven pricing passes between 08:09 and 13:37 UTC, each wave designed after the previous pass's dev91 numbers, so the last row is the top of "
           f"an adaptive series. Its 95% interval is {R11['series_ci'][0]:.0f} to {R11['series_ci'][1]:.0f}; the last step (+2.6 Elo) is far inside it. The selection board that "
           "locked the later entrants is weaker than it looked: the base neural checkpoints were pre-trained on the SEARCH rows of 73 of its 88 clean contexts. Variants priced "
           "in the same pass but not locked are not shown. Pass 5 also locked an entrant at 1847.9 that was withdrawn as not 0% dev91 (transitive GP anchors), and an "
           "in-corpus entrant; neither is shown, so 1842.0 is not the highest locked number. (Accounting by the reviewing agent, 2026-11-15 14:06 UTC.)", "cap"))
S.append(P("Staged residual GP (S9, the owner's experiment): closed, no arm passes", "h2"))
rows = [["Block", "Built\nby", "Scored\nby", "Arms\npassing", "Gain statistic T,\neligible arms", "Two-seed\nbag T", "Stack Elo:\nsingle runs", "Stack Elo:\neligible arms"]]
for r in R11["s9"]:
    rows.append([P(r["label"], "cell"), r["builder"], r["scorer"], f"{r['n_r1']} of {r['n_eligible']}", f"{r['t_min']:+.3f} to {r['t_max']:+.3f}",
                 f"{r['runbag_t']:+.3f}", f"{min(r['elo_a'], r['elo_b']):.0f} to {max(r['elo_a'], r['elo_b']):.0f}", f"{r['elo_worst_eligible']:.0f} to {r['elo_best_eligible']:.0f}"])
S.append(table(rows, [2.35 * inch, 0.6 * inch, 0.6 * inch, 0.65 * inch, 1.05 * inch, 0.6 * inch, 0.8 * inch, 0.8 * inch], align_right=(3, 4, 5, 6, 7), font=6.7, pad=1.6))
S.append(P("A chain fits each GP run to the residual of the runs before it. T is the registered gain over the first control seed, in utility; an arm passes only if it beats both "
           "control seeds after correction, and the double-budget single run must not dominate it. No arm passes on any block, at 40 or 160 generations, with either anchor, "
           "from either builder; each builder's blocks were scored by another agent. The stack without any of these columns is at 1788 Elo on this board. The one positive "
           "sign is averaging two independent seed runs (+0.03 to +0.05), which vanishes with the hybrid anchor. Each block is 80 contexts of the design board P201B. The 160-generation tier of the second builder ran under a recorded stop and "
           "restart; no arm is near the line, and the registered load-matched second read reproduces the main read exactly (descriptive).", "cap"))
L45 = R11["line45"]
S.append(P("Per-table fine-tuning inside a stack: absorbed", "h2"))
S.append(P(f"<b>Added to a stack, the fine-tune of section 15 wins on {L45['w']} of 10 contexts; its rule needed 7.</b> Fifty cross-fitted fine-tunes (ten contexts, five folds) "
           "gave a column whose SEARCH-row values come from a model that never saw the row. In a seven-column stand-in stack (frozen ICL, ridge, nearest neighbours, boosted "
           f"tree, one out-of-fold GP run) it changes fresh-row utility by {L45['du']:+.2f} in the mean (sign-flip p = {L45['p']:.2f}): "
           + ", ".join(f"{r['family']} {r['du']:+.2f}" for r in L45["by_family"][:3]) + f", and between {L45['by_family'][-1]['du']:+.2f} and {L45['by_family'][3]['du']:+.2f} on the "
           f"other seven. Without the cross-fit the stack puts 92% of its weight on the column and loses on {L45['naive_l']} of 10. A replay of the two large gains on the frozen "
           f"dev91 board is worth {R11['capgap'][1]['elo'] - R11['capgap'][0]['elo']:+.1f} Elo, and on those two tables the fair champion already scores higher on audit rows "
           "than the fine-tuned stand-in stack does on fresh rows. A gated version was struck before any GPU was spent.", "body"))
FD = R11["fielding"]; TE = R11["tailcal_elo"]
S.append(P("Selection, fusion and recalibration of what we hold: closed (Track S, design board of 121 contexts)", "h2"))
S += bullets([
    f"<b>A fielding floor finds nothing.</b> A rule that leaves the stack for a classical candidate only when a lower bound on SEARCH rows says so leaves it on 2 of {FD['n_ctx']} "
    f"contexts, both exact-law tables, and on {R11['nolook_max_leaves']} of 91 dev91 contexts under all {R11['nolook_configs']} settings (SEARCH rows only, nothing scored). On the 18 "
    "design contexts where a candidate did beat the stack on audit rows, that candidate was worse on SEARCH rows every time. The +73 Elo floor ceiling of section 15 was "
    "audit sampling on heavy tails, not a selection error.",
    "<b>Nothing we hold adds to the stack on ordinary contexts.</b> The best blend weight of every classical column is zero on most contexts; our ICL columns are "
    "0.95-correlated with the stack and about 0.77 utility weaker.",
    f"<b>The stack is calibrated.</b> The slope of the target on the stack's prediction is 1.01 on audit rows. Its squared error sits in the largest targets (median share of "
    f"the top tenth {FD['top_share'] * 100:.0f}%; over half on {FD['n_top_half']} of {FD['n_ctx']} contexts), but as variance, not bias: linear recalibration scores "
    f"{TE['LIN']:.0f} and isotonic {TE['ISO']:.0f} against {TE['V0']:.0f} for the stack as it is.",
], "bullet_s")
S.append(P("Open decisions for the owner", "h2"))
S += bullets([
    "Whether an entrant expected to gain about 5 Elo is worth one of the two remaining dev91 looks, and which board this wave has not read confirms it.",
    "An external model as a training-time teacher for our own head (inference stays ours only), or a separately labelled hybrid line with open models as stack members. "
    "These are the only levers priced above 30 Elo.",
], "bullet_s")
S.append(P("Sources", "h2"))
S.append(P("<font face='DVM' size='6.6'>~/jope_plan/evidence/H450/board_post_h448/dev91-audit-v1/{BOARD.json, ladder.json, scores/*.jsonl}</font> (frozen field, per-context scores)<br/>"
           "<font face='DVM' size='6.6'>~/jope_plan/evidence/H455/*MATRIX*.json, H454/*MATRIX*.json, H453/*MATRIX*.json, H452/**/*.json</font> (scoring matrices)<br/>"
           "<font face='DVM' size='6.6'>~/jope_plan/evidence/H450/diag_results/r4_g*/H450_DEV91.json, H450/r4_traj_final.tsv</font> (r4 epochs)<br/>"
           "<font face='DVM' size='6.6'>~/jope_plan/evidence/H449/diag_results/H449_UNFAIR_DEV91.json</font> (H449 fair and in-corpus arms)<br/>"
           "<font face='DVM' size='6.6'>~/jope-data/h458-quarantine/probe/*.json</font> (H458 SEARCH-only probes)<br/>"
           "<font face='DVM' size='6.6'>~/jope_plan/evidence/H458/{LOOK_LEDGER.jsonl, trackD/look2/SCORE.json, trackD/probe/*/SCORE.json}</font> (H458 look 2, receipt sha256 "
           f"<font face='DVM' size='6.6'>{L2['receipt_sha256'][:16]}</font>, and the fresh-row arms)<br/>"
           "<font face='DVM' size='6.6'>~/jope_plan/evidence/H455/H455_TOP_ARMS_PER_CONTEXT_U.jsonl, ~/jope-data/h458s-quarantine/leaderboard/inputs_*/B_h451_h457.jsonl</font> "
           "(per-context utilities of the best arm and the fair-closure arm)<br/>"
           "<font face='DVM' size='6.6'>~/jope_plan/evidence/H456/H456_STAGE3_*.{json,jsonl}, H462/M6_SEED_BAGGING_PRICE.json, H458/trackD/probe/FRESH_*/SCORE.json, "
           "~/jope-data/h458-quarantine/nextwave/*.json</font> (section 15)<br/>"
           "<font face='DVM' size='6.6'>~/jope-data/h458s-quarantine/{s9_stack/*.log, s10/*.json}, ~/jope_plan/evidence/H462/M6_*_PRICE*.json, "
           "H458/trackD/probe/FRESH_STACK_XFIT_OWN_FINETUNE/SCORE.json</font> (section 16)<br/>"
           f"Collected bundle: <font face='DVM' size='6.6'>~/jope-data/h458-quarantine/report/elo_report_bundle.json</font> ({T['n_arm_rows'] - 7:,} arm entries from "
           f"{T['n_files'] - 1} files, generated {T['generated_utc']}), plus the 7 arm entries of r4 epoch 48 read from its result file. Board sha256 <font face='DVM' size='6.6'>{T['board']['sha256'][:16]}</font>. All paths are on host xbabe2. "
           "Nothing was re-scored for this report and no audit label was read; the H458 look was scored once by Track S's scorer under the look ledger.", "note"))

doc.build(S)
print("wrote", OUT, os.path.getsize(OUT))
