from __future__ import annotations

import csv
import io

import db


def build_csv_report(day: str | None = None) -> str:
    report_date, start, end = db.get_shift_bounds(day)
    return build_csv_report_range(start, end, report_date)


def build_csv_report_range(start: str, end: str, report_date: str | None = None) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["report_date", report_date or start[:10]])
    writer.writerow([])
    writer.writerow(["time", "camera", "zone", "type", "status", "confidence", "reaction_seconds", "description"])
    for event in db.list_events(date_from=start, date_to=end):
        writer.writerow([
            event.detected_at,
            event.camera_name,
            event.zone,
            event.type,
            event.status,
            round(event.confidence * 100),
            event.reaction_seconds or "",
            event.description,
        ])
    return output.getvalue()


def build_pdf_report(day: str | None = None) -> bytes:
    _report_date, start, end = db.get_shift_bounds(day)
    return build_pdf_report_range(start, end)


def build_pdf_report_range(start: str, end: str) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    events = db.list_events(date_from=start, date_to=end)
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    _width, height = A4
    y = height - 50
    pdf.setFont("Helvetica", 16)
    pdf.drawString(40, y, "AI Video Monitoring shift report")
    y -= 28
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, f"Period: {start} - {end}")
    y -= 28
    for event in events[:30]:
        line = f"{event.detected_at} | {event.camera_name} | {event.type} | {event.status}"
        pdf.drawString(40, y, line[:120])
        y -= 16
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 10)
            y = height - 50
    pdf.save()
    return buffer.getvalue()
