"""
Reporting Service
Generates professional PDF job cards for service tickets.
"""
import io
import logging
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from ..core.schemas import ServiceTicket

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates PDF reports."""
    
    @staticmethod
    def generate_job_card(ticket: ServiceTicket) -> bytes:
        """Create a PDF job card from a ticket."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1E88E5'),
            spaceAfter=20
        )
        
        header_style = ParagraphStyle(
            'Header',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#424242'),
            spaceAfter=10
        )
        
        elements = []
        
        # Title
        elements.append(Paragraph("Automotive Diagnostic Job Card", title_style))
        elements.append(Paragraph(f"Ticket ID: {ticket.request_id}", styles['Normal']))
        elements.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
        elements.append(Spacer(1, 20))
        
        # Complaint Section
        elements.append(Paragraph("1. Customer Complaint", header_style))
        elements.append(Paragraph(f"<i>Current Issue:</i> {ticket.original_complaint}", styles['Normal']))
        elements.append(Spacer(1, 15))
        
        # Diagnosis Section
        elements.append(Paragraph("2. System Diagnosis", header_style))
        
        if ticket.classification and ticket.classification.primary_intent:
            intent = ticket.classification.primary_intent
            diag_data = [
                ['System', intent.system],
                ['Component', intent.component],
                ['Failure Mode', intent.failure_mode],
                ['Confidence', f"{int(intent.confidence * 100)}%"]
            ]
            
            t = Table(diag_data, colWidths=[150, 300])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F5')),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0'))
            ]))
            elements.append(t)
        else:
            elements.append(Paragraph("Diagnosis pending or ambiguous.", styles['Normal']))
            
        elements.append(Spacer(1, 15))
        
        # Action Plan
        elements.append(Paragraph("3. Recommended Action", header_style))
        if ticket.normalization:
            action = ticket.normalization.suggested_action
            elements.append(Paragraph(action, styles['Normal']))
        else:
            elements.append(Paragraph("Inspect identified component.", styles['Normal']))
            
        elements.append(Spacer(1, 30))
        
        # Disclaimer
        disclaimer = """
        <b>DISCLAIMER:</b> This diagnosis is AI-generated and should be verified by a certified mechanic 
        before starting any repairs. The service center assumes no liability for incorrect diagnosis.
        """
        elements.append(Paragraph(disclaimer, styles['Italic']))
        
        # Build PDF
        doc.build(elements)
        
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes

# Singleton
_generator = None

def get_report_generator() -> ReportGenerator:
    global _generator
    if _generator is None:
        _generator = ReportGenerator()
    return _generator
