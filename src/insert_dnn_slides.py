from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_VERTICAL_ANCHOR
from pptx.util import Inches, Pt


DECK_PATH = "reports/STAT5243 Project4.pptx"

BG = RGBColor(255, 255, 255)
WHITE = RGBColor(15, 23, 42)
MUTED = RGBColor(71, 85, 105)
ACCENT = RGBColor(56, 189, 248)
ACCENT_SOFT = RGBColor(14, 165, 233)
CARD = RGBColor(241, 245, 249)
CARD_2 = RGBColor(248, 250, 252)
GOOD = RGBColor(34, 197, 94)
WARN = RGBColor(249, 115, 22)
PURPLE = RGBColor(168, 85, 247)
TEAL = RGBColor(45, 212, 191)

SLIDE_W = Inches(10)
SLIDE_H = Inches(5.625)


def clear_slide(slide):
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)


def set_background(slide, color=BG):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def style_run(run, size, bold=False, color=WHITE, font_name="Aptos"):
    run.font.name = font_name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_textbox(slide, left, top, width, height, text="", size=20, bold=False,
                color=WHITE, align=PP_ALIGN.LEFT, fill=None, line=None,
                margin=0.08, font_name="Aptos"):
    box = slide.shapes.add_textbox(left, top, width, height)
    if fill is not None:
        box.fill.solid()
        box.fill.fore_color.rgb = fill
    else:
        box.fill.background()
    if line is not None:
        box.line.color.rgb = line
        box.line.width = Pt(1.4)
    else:
        box.line.fill.background()
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = MSO_VERTICAL_ANCHOR.TOP
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    style_run(run, size=size, bold=bold, color=color, font_name=font_name)
    return box


def add_paragraph(tf, text, size=18, color=WHITE, bold=False, level=0,
                  bullet=False, align=PP_ALIGN.LEFT, space_after=6):
    p = tf.add_paragraph()
    p.alignment = align
    p.level = level
    p.space_after = Pt(space_after)
    if bullet:
        p.text = text
    else:
        run = p.add_run()
        run.text = text
        style_run(run, size=size, bold=bold, color=color)
    if bullet:
        for run in p.runs:
            style_run(run, size=size, bold=bold, color=color)
    return p


def add_title(slide, title):
    return add_textbox(slide, Inches(0.45), Inches(0.22), Inches(8.6), Inches(0.55),
                       title, size=26, bold=True, color=WHITE)


def add_counter(slide, n):
    return add_textbox(slide, Inches(8.7), Inches(5.08), Inches(1.0), Inches(0.28),
                       f"Slide {n} / 4", size=11, color=MUTED, align=PP_ALIGN.RIGHT)


def add_arrow(slide, x1, y1, x2, y2, color=ACCENT, width=2.0):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    line.line.color.rgb = color
    line.line.width = Pt(width)
    line.line.end_arrowhead = True
    return line


def add_round_box(slide, left, top, width, height, fill=CARD, line=ACCENT, radius_shape=True):
    shape_type = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius_shape else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.8)
    return shape


def add_table(slide, rows, cols, left, top, width, height, data, header_fill=ACCENT_SOFT):
    table = slide.shapes.add_table(rows, cols, left, top, width, height).table
    col_w = width // cols
    for c in range(cols):
        table.columns[c].width = col_w
    row_h = height // rows
    for r in range(rows):
        table.rows[r].height = row_h
    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = str(data[r][c])
            cell.fill.solid()
            cell.fill.fore_color.rgb = header_fill if r == 0 else CARD_2
            cell.margin_left = Inches(0.05)
            cell.margin_right = Inches(0.05)
            cell.margin_top = Inches(0.03)
            cell.margin_bottom = Inches(0.03)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c == 0 else PP_ALIGN.LEFT
            for run in p.runs:
                style_run(run, size=11 if r == 0 else 10, bold=(r == 0), color=WHITE)
    return table


def slide1(slide):
    set_background(slide)
    add_title(slide, "Predicting Customer Review Score from Order Data")

    # Left question card
    add_textbox(slide, Inches(0.5), Inches(1.0), Inches(3.45), Inches(1.05),
                "The Question", size=17, bold=True, fill=CARD, line=ACCENT)
    qbox = add_textbox(slide, Inches(0.5), Inches(1.46), Inches(3.45), Inches(1.35),
                       "", size=16, fill=CARD_2, line=ACCENT)
    tf = qbox.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = ("Given everything known at dispatch time — logistics, product, "
                "payment — can we predict the star rating an order will receive?")
    style_run(run, size=16, color=WHITE)

    # Dataset callout
    ds = add_textbox(slide, Inches(0.5), Inches(3.0), Inches(3.45), Inches(1.55),
                     "", fill=CARD, line=ACCENT)
    tf = ds.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Dataset snapshot"
    style_run(run, size=16, bold=True, color=ACCENT)
    add_paragraph(tf, "108,911 Brazilian e-commerce orders (Olist)", size=14, bullet=True, color=WHITE)
    add_paragraph(tf, "80 / 20 stratified train / test split", size=14, bullet=True, color=WHITE)
    add_paragraph(tf, "Target: review_score (1–5 ★)", size=14, bullet=True, color=WHITE)

    # Feature group table
    add_textbox(slide, Inches(4.2), Inches(1.0), Inches(5.2), Inches(0.42),
                "Feature groups used by the DNN", size=17, bold=True, color=WHITE)
    data = [
        ["Group", "Examples"],
        ["Logistics", "delay_days, delivery_days, distance_km, freight_ratio,\nseller_recent_delay_avg"],
        ["Product", "volume_cm³, weight, category (~70 types), photos"],
        ["Payment", "type, installments, value"],
        ["Geography", "seller_state, customer_state (one-hot → 145 total features)"],
    ]
    add_table(slide, 5, 2, Inches(4.2), Inches(1.45), Inches(5.1), Inches(2.75), data)

    add_textbox(slide, Inches(4.2), Inches(4.45), Inches(5.1), Inches(0.55),
                "57.5% of orders are 5★ — severe class imbalance shapes the task",
                size=13, color=MUTED, fill=None, line=None)
    add_counter(slide, 1)


def slide2(slide):
    set_background(slide)
    add_title(slide, "Feed-Forward MLP")

    add_textbox(slide, Inches(0.55), Inches(0.86), Inches(5.0), Inches(0.36),
                "Architecture + Training Summary", size=17, bold=True)
    box = add_textbox(slide, Inches(0.55), Inches(1.22), Inches(8.55), Inches(3.78),
                      "", fill=CARD, line=ACCENT, margin=0.12)
    tf = box.text_frame
    tf.clear()

    bullet_blocks = [
        "• 145 input features — logistics, product, payment, geography\n"
        "  (one-hot encoded categoricals + pre-engineered numerics)",
        "• 4 hidden layers, width 90 — each block:\n"
        "  Linear → BatchNorm → ReLU → Dropout (p = 0.32)",
        "• Output: single scalar, no activation — direct regression on review score (1–5)\n"
        "  38,521 trainable parameters total",
        "• Trained with AdamW + cosine learning-rate decay\n"
        "  Early stopping: patience 10 epochs, best weights restored",
        "• Hyperparameter search: Optuna Bayesian optimisation\n"
        "  20 trials × 3-fold CV — objective: mean fold RMSE\n"
        "  Best trial: lr = 9.3 × 10⁻³, dropout = 0.32, 4 layers × width 90",
        "• Test set (21,783 orders): RMSE = 1.22 | MAE = 0.95 | R² = 0.18\n"
        "  Predictions monotonically ordered across all 5 star buckets\n"
        "  (mean prediction 3.42 for 1★ → 4.21 for 5★)",
    ]

    for i, text in enumerate(bullet_blocks):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(7)
        for run in p.runs:
            style_run(run, size=12.6 if i < 4 else 12.2, color=WHITE)
    add_counter(slide, 2)


def slide3(slide):
    set_background(slide)
    add_title(slide, "Bayesian Search + K-Fold CV")

    add_textbox(slide, Inches(0.45), Inches(0.95), Inches(3.8), Inches(0.35),
                "Training loop", size=17, bold=True)

    steps = [
        "For each Optuna trial",
        "For each of K = 3 folds",
        "Fit StandardScaler on fold train",
        "Train with AdamW + cosine LR",
        "Early stop (patience = 10)",
        "Objective = mean fold RMSE",
        "Prune if worse than median",
        "Report best trial",
    ]
    y = Inches(1.33)
    for idx, step in enumerate(steps):
        width = Inches(3.35 if idx in (0, 1, 5, 7) else 3.05)
        left = Inches(0.55 if idx in (0, 5, 7) else 0.78)
        shape = add_round_box(slide, left, y, width, Inches(0.34),
                              fill=CARD_2 if idx not in (0, 7) else CARD, line=ACCENT)
        tf = shape.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = step
        style_run(run, size=12.5, color=WHITE, bold=(idx in (0, 7)))
        if idx < len(steps) - 1:
            add_arrow(slide, Inches(2.2), y + Inches(0.34), Inches(2.2), y + Inches(0.46))
        y += Inches(0.5)

    add_textbox(slide, Inches(4.55), Inches(0.95), Inches(4.4), Inches(0.35),
                "Search space", size=17, bold=True)
    table_data = [
        ["Parameter", "Range"],
        ["Layers", "2 – 5"],
        ["Width", "64 – 512 (log)"],
        ["Dropout", "0.0 – 0.5"],
        ["Learning rate", "1e-4 – 1e-2 (log)"],
        ["Weight decay", "1e-5 – 1e-2 (log)"],
        ["Batch size", "1024 / 2048 / 4096"],
        ["Activation", "ReLU / GELU / ELU"],
    ]
    add_table(slide, 8, 2, Inches(4.55), Inches(1.35), Inches(4.2), Inches(2.8), table_data)

    callout = add_textbox(slide, Inches(4.55), Inches(4.35), Inches(4.2), Inches(0.78),
                          "", fill=CARD, line=ACCENT)
    tf = callout.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Best config (trial 16 / 20): "
    style_run(run, size=14, bold=True, color=ACCENT)
    run = p.add_run()
    run.text = "4 layers × width 90, ReLU, dropout 0.32, lr = 9.3 × 10⁻³"
    style_run(run, size=14, color=WHITE)
    p = tf.add_paragraph()
    run = p.add_run()
    run.text = "Then retrained once on the full 80% train split."
    style_run(run, size=13, color=MUTED)
    add_counter(slide, 3)


def slide4(slide):
    set_background(slide)
    add_title(slide, "Performance on Held-Out Test Set (21,783 orders)")

    metrics = [("RMSE", "1.22"), ("MAE", "0.95"), ("R²", "0.18")]
    lefts = [Inches(0.55), Inches(3.42), Inches(6.29)]
    colors = [ACCENT, GOOD, PURPLE]
    for (label, value), left, color in zip(metrics, lefts, colors):
        card = add_round_box(slide, left, Inches(1.0), Inches(2.45), Inches(1.0),
                             fill=CARD, line=color)
        tf = card.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = value
        style_run(run, size=24, bold=True, color=WHITE)
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = label
        style_run(run, size=13, color=MUTED)

    add_textbox(slide, Inches(0.55), Inches(2.35), Inches(4.3), Inches(0.35),
                "Per-star prediction summary", size=17, bold=True)
    per_star = [
        ["True ★", "Mean predicted", "n"],
        ["1★", "3.42", "2,480"],
        ["2★", "3.74", "731"],
        ["3★", "3.98", "1,833"],
        ["4★", "4.14", "4,199"],
        ["5★", "4.21", "12,540"],
    ]
    add_table(slide, 6, 3, Inches(0.55), Inches(2.78), Inches(4.25), Inches(1.95), per_star)

    takeaway = add_textbox(slide, Inches(5.15), Inches(2.35), Inches(4.1), Inches(2.45),
                           "", fill=CARD, line=ACCENT)
    tf = takeaway.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = "Main takeaway"
    style_run(run, size=16, bold=True, color=ACCENT)
    p = tf.add_paragraph()
    run = p.add_run()
    run.text = "Predictions are monotonically ordered across all five buckets."
    style_run(run, size=16, bold=True, color=WHITE)
    p.space_after = Pt(8)
    p = tf.add_paragraph()
    run = p.add_run()
    run.text = ("So the DNN learned the direction of the logistics signal, "
                "but low R² shows a lot of review noise still comes from factors "
                "we do not observe, like product quality or personal expectations.")
    style_run(run, size=15, color=WHITE)
    add_counter(slide, 4)


def main():
    prs = Presentation(DECK_PATH)

    while len(prs.slides) < 18:
        prs.slides.add_slide(prs.slide_layouts[10])

    target_slides = [prs.slides[i] for i in range(14, 18)]
    for slide in target_slides:
        clear_slide(slide)

    slide1(target_slides[0])
    slide2(target_slides[1])
    slide3(target_slides[2])
    slide4(target_slides[3])

    prs.save(DECK_PATH)


if __name__ == "__main__":
    main()
