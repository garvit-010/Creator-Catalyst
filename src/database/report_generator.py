"""
PDF Report Generator for Creator Catalyst
Generates professional "Campaign Summary" reports for processed videos.
"""

import json
from datetime import datetime
from pathlib import Path
from fpdf import FPDF
from src.database.database import Video, ContentOutput, GroundingReport, Database

class CampaignReportGenerator:
    """Handles PDF generation for campaign summaries."""
    
    def __init__(self, db: Database, output_dir: str = "data/reports"):
        self.db = db
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, video_id: int) -> str:
        """
        Generate a professional PDF report for a video.
        Returns the path to the generated PDF.
        """
        # Fetch data
        video = self.db.get_video(video_id)
        if not video:
            raise ValueError(f"Video with ID {video_id} not found")
            
        outputs = self.db.get_content_by_video(video_id)
        grounding = self.db.get_grounding_report(video_id)
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_fill_color(41, 128, 185)  # Professional Blue
        pdf.rect(0, 0, 210, 40, 'F')
        
        pdf.set_font("helvetica", "B", 24)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 20, "CAMPAIGN SUMMARY", ln=True, align="C")
        
        pdf.set_font("helvetica", "", 12)
        pdf.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        
        pdf.ln(20)
        
        # Video Overview
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "🎥 Video Overview", ln=True)
        pdf.set_font("helvetica", "", 12)
        
        col_width = 45
        pdf.cell(col_width, 10, "Filename:", b=0)
        pdf.cell(0, 10, video.filename, ln=True)
        
        pdf.cell(col_width, 10, "Platform:", b=0)
        pdf.cell(0, 10, video.platform, ln=True)
        
        pdf.cell(col_width, 10, "Duration:", b=0)
        duration = f"{video.duration_seconds}s" if video.duration_seconds else "N/A"
        pdf.cell(0, 10, duration, ln=True)
        
        pdf.cell(col_width, 10, "Uploaded:", b=0)
        pdf.cell(0, 10, video.uploaded_at, ln=True)
        
        pdf.ln(10)
        
        # Grounding & Accuracy
        if grounding:
            pdf.set_font("helvetica", "B", 16)
            pdf.cell(0, 10, "🛡️ Accuracy & Fact-Grounding", ln=True)
            pdf.set_font("helvetica", "", 12)
            
            pdf.cell(col_width, 10, "Total Claims:", b=0)
            pdf.cell(0, 10, str(grounding.total_claims), ln=True)
            
            pdf.cell(col_width, 10, "Verified:", b=0)
            pdf.set_text_color(39, 174, 96)  # Green
            pdf.cell(0, 10, f"{grounding.verified_claims} ({(grounding.verified_claims/grounding.total_claims*100):.1f}%)" if grounding.total_claims > 0 else "N/A", ln=True)
            pdf.set_text_color(0, 0, 0)
            
            pdf.ln(10)
            
        # Content Strategy
        pdf.set_font("helvetica", "B", 16)
        pdf.cell(0, 10, "📝 Content Strategy Breakdown", ln=True)
        pdf.set_font("helvetica", "", 12)
        
        for output in outputs:
            pdf.set_font("helvetica", "B", 13)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(0, 10, f"Type: {output.content_type.replace('_', ' ').title()}", ln=True, fill=True)
            
            pdf.set_font("helvetica", "", 11)
            # Clean content for PDF
            content_text = output.content.encode('latin-1', 'replace').decode('latin-1')
            pdf.multi_cell(0, 7, content_text[:500] + ("..." if len(content_text) > 500 else ""))
            pdf.ln(5)

        # Footer
        pdf.set_y(-30)
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(128, 128, 128)
        pdf.cell(0, 10, "Powered by Creator Catalyst - ROI & Strategy Insights", align="C")
        
        # Save file
        safe_name = "".join(x for x in video.filename if x.isalnum() or x in "._- ")
        report_path = self.output_dir / f"Report_{video_id}_{safe_name}.pdf"
        pdf.output(str(report_path))
        
        return str(report_path)

def get_report_generator():
    """Helper to get generator instance."""
    from src.database.database import Database
    db = Database()
    return CampaignReportGenerator(db)
