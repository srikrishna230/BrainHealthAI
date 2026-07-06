from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import os


def generate_pdf(result):

    # Create reports folder if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate("reports/Brain_Report.pdf")

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph("<b>Brain Health Monitoring Report</b>", styles["Title"])
    )

    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(
        Paragraph(f"Mental State : {result['Prediction']}", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Confidence : {result['Confidence']:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Stress : {result['Stress']:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Anxiety : {result['Anxiety']:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Depression : {result['Depression']:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Happiness : {result['Happiness']:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Attention : {result['Attention']:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Brain Health : {result['BrainHealth']:.2f}%", styles["BodyText"])
    )

    story.append(
        Paragraph(f"Neutral : {result['Neutral']:.2f}%", styles["BodyText"])
    )

    # Build the PDF (IMPORTANT: inside the function)
    doc.build(story)

    return "reports/Brain_Report.pdf"