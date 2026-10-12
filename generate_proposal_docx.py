import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def create_proposal_document(filename="Project_Proposal_News_Recommender.docx"):
    doc = Document()
    
    # Page setup (1-inch margins)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Palette
    primary_color = RGBColor(24, 76, 120)    # Deep Navy
    secondary_color = RGBColor(41, 128, 185) # Accent Blue
    accent_green = RGBColor(39, 174, 96)     # Cost/Success Green
    text_dark = RGBColor(44, 62, 80)         # Charcoal

    # Title
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run_title = title.add_run("Intelligent News Recommendation Engine")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = primary_color
    title.paragraph_format.space_after = Pt(2)

    # Subtitle
    sub = doc.add_paragraph()
    run_sub = sub.add_run("Prototype Architecture, Publish-Time Execution Workflow & Cost Optimization Blueprint")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.color.rgb = secondary_color
    sub.paragraph_format.space_after = Pt(12)

    # Metadata Box
    meta_table = doc.add_table(rows=1, cols=1)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_cell = meta_table.cell(0, 0)
    set_cell_background(meta_cell, "F4F6F7")
    set_cell_margins(meta_cell, top=140, bottom=140, left=200, right=200)
    
    meta_p = meta_cell.paragraphs[0]
    meta_p.paragraph_format.space_after = Pt(0)
    m_run = meta_p.add_run("DOCUMENT METADATA & PURPOSE\n")
    m_run.font.bold = True
    m_run.font.size = Pt(9.5)
    m_run.font.color.rgb = primary_color
    
    m_details = meta_p.add_run(
        "• Target Audience: Engineering Leads, Product Managers, Editorial Tech Team\n"
        "• Strategic Goal: Present working prototype findings, execution lifecycle, and modular features for main product rollout\n"
        "• Key Focus: Zero-live-inference cost architecture, windowed similarity matching, and production PostgreSQL + pgvector integration"
    )
    m_details.font.size = Pt(9.5)
    m_details.font.color.rgb = text_dark

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # Helper functions
    def add_heading_1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(16)
        h.paragraph_format.space_after = Pt(4)
        run = h.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(15)
        run.font.bold = True
        run.font.color.rgb = primary_color
        return h

    def add_heading_2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(2)
        run = h.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(12)
        run.font.bold = True
        run.font.color.rgb = secondary_color
        return h

    def add_body(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(10.5)
        run.font.color.rgb = text_dark
        return p

    def add_bullet(bold_prefix, text):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        bp = p.add_run(bold_prefix)
        bp.font.name = "Calibri"
        bp.font.size = Pt(10)
        bp.font.bold = True
        bp.font.color.rgb = text_dark
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(10)
        run.font.color.rgb = text_dark
        return p

    # 1. Executive Summary
    add_heading_1("1. Executive Summary")
    add_body(
        "News media publishing operates under unique constraints that break traditional e-commerce recommendation systems. "
        "News stories have a short lifecycle (24–48 hours), break unexpectedly, and must immediately match reader intent without "
        "waiting for thousands of clicks to accumulate. This prototype introduces a robust, multi-stage hybrid recommendation "
        "engine that combines modern NLP semantic embeddings, keyword retrieval (BM25), cross-encoder neural reranking, "
        "and editorial domain heuristics."
    )
    add_body(
        "A primary concern for the main engineering team when evaluating neural rerankers (Cross-Encoders) is live computational "
        "overhead and cloud hosting expenses. This document details an enterprise-grade execution plan: moving model inference "
        "entirely to an asynchronous publish-time workflow. By generating embeddings and reranking against time-windowed candidates "
        "offline, live web traffic incurs ZERO machine learning inference cost."
    )

    # 2. Detailed Execution Plan: Publish-Time Ingestion & Offline Reranking
    add_heading_1("2. Detailed Execution Plan: Publish-Time Ingestion & Offline Reranking")
    add_body(
        "To achieve maximum precision without burdening live web servers, the proposed production architecture "
        "decouples recommendation generation from user request serving. Here is the end-to-end execution workflow:"
    )
    
    add_bullet("Step 1 (Publisher Action): ", "A journalist or editor creates and publishes an article in the CMS (Content Management System).")
    add_bullet("Step 2 (Asynchronous Webhook): ", "Upon hitting 'Publish', the CMS fires a lightweight webhook payload to an asynchronous ingestion queue (e.g., Celery, Redis Queue, or AWS SQS). The live publishing UI returns instantly to the editor.")
    add_bullet("Step 3 (Embedding & pgvector Storage): ", "The background worker parses and cleans the article text (headline, lead paragraph, section). It passes the text through the bi-encoder (all-MiniLM-L6-v2) to generate a 384-dimensional vector, storing it directly into the PostgreSQL 'articles' table via the pgvector extension.")
    add_bullet("Step 4 (Time-Windowed Candidate Retrieval): ", "Instead of scanning the entire historical database of hundreds of thousands of articles, the worker queries pgvector using a time-bounded window (e.g., last 7 to 30 days) and section filtering. This retrieves the top 20–30 closest candidates in under 5ms, keeping compute tight and relevance fresh.")
    add_bullet("Step 5 (Offline Cross-Encoder Reranking): ", "The worker feeds the new article and the 20–30 windowed candidates into the Cross-Encoder (ms-marco-MiniLM-L-6-v2). Joint cross-attention accurately ranks candidates by true editorial nuance. Calibrated boosts for same-section (8%), subsection (5%), and organic pageviews are factored in.")
    add_bullet("Step 6 (Precomputation Persistence): ", "The worker writes the resulting top 5–10 recommended article IDs directly into PostgreSQL (ALTER TABLE articles ADD COLUMN related_article_ids INT[]) and warms the Redis cache (SET related:article:{id} '[101, 204, 305]').")
    add_bullet("Step 7 (Live User Traffic - Zero ML Cost): ", "When millions of users read the article on the website or mobile app, the backend performs a single O(1) cache lookup or indexed SQL query. Live request latency is < 5ms with ZERO neural model execution.")

    # 3. Dedicated Cost Optimization Blueprint
    add_heading_1("3. Comprehensive Cost Optimization Blueprint")
    add_body(
        "For a high-traffic news platform, cost optimization is vital. The prototype architecture incorporates several "
        "cost-saving strategies designed to keep infrastructure bills under $30–$50/month:"
    )
    add_bullet("1. Time-Windowed Candidate Filtering: ", "News is perishable. Restricting the similarity search to articles published in the last 7–30 days reduces the candidate pool by 95%+, drastically cutting vector comparison math and indexing memory.")
    add_bullet("2. Offline Precomputation vs. Live Inference: ", "Eliminates the need for 24/7 dedicated GPU instances (e.g., AWS g4dn/g5 instances costing $500–$1,200+/month). The background worker runs comfortably on standard, low-cost CPU instances.")
    add_bullet("3. Model Quantization (INT8 & ONNX Runtime): ", "The embedding and cross-encoder models can be exported to ONNX format with 8-bit integer quantization (INT8) or using FlashRank. This cuts model memory usage from 500MB to ~70MB and triples CPU throughput.")
    add_bullet("4. Edit Debouncing: ", "Editors frequently make minor headline or typo corrections within minutes of publishing. Ingestion webhooks are debounced by 60–90 seconds to prevent redundant reranking passes.")
    add_bullet("5. PostgreSQL Table Partitioning: ", "Partition the 'articles' table by month. HNSW vector indexes only need to remain in active RAM for the latest active months, allowing older archive data to stay on standard disk storage.")
    add_bullet("6. Edge/CDN Caching: ", "Because related article lists for an article are precomputed, API responses can carry HTTP cache headers (Cache-Control: public, max-age=600). 95%+ of recommendation requests can be absorbed by Cloudflare or CloudFront without even hitting the database.")

    # 4. Strategic Comparison: Prototype vs. AWS Personalize
    add_heading_1("4. Strategic Assessment: In-House Engine vs. AWS Personalize")
    add_body(
        "When comparing this hybrid approach against commercial cloud recommendation engines (e.g., AWS Personalize), "
        "the advantages for news media are distinct:"
    )

    t = doc.add_table(rows=6, cols=3)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Evaluation Metric", "AWS Personalize", "In-House Hybrid Engine (Our Prototype)"]
    for i, h in enumerate(headers):
        cell = t.cell(0, i)
        set_cell_background(cell, "184C78")
        set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.font.bold = True
        run.font.size = Pt(9.5)
        run.font.color.rgb = RGBColor(255, 255, 255)

    rows_data = [
        ("Breaking News Cold-Start", "Poor (requires accumulated click logs; cannot recommend new articles effectively)", "Instant (0 ms delay; embeddings & BM25 available the moment of publish)"),
        ("Live Server ML Cost", "High (pay-per-TPS inference campaigns running 24/7)", "Zero ($0 live inference; precomputed at publish time)"),
        ("Editorial Control & Tuning", "Black Box (fixed algorithm weights; cannot inject section rules)", "100% Tunable (precise control over section, subsection & popularity boosts)"),
        ("Named Entity Recognition", "Limited to shallow metadata tags", "Deep semantic + lexical matching on article headlines and bodies"),
        ("Estimated Monthly Cost", "High ($300 - $1,500+/mo for dedicated inference campaigns)", "Extremely Low ($15 - $40/mo standard Postgres/FastAPI instance)")
    ]

    for row_idx, data in enumerate(rows_data, start=1):
        for col_idx, text in enumerate(data):
            cell = t.cell(row_idx, col_idx)
            bg = "FFFFFF" if row_idx % 2 == 1 else "F8F9FA"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            run = p.add_run(text)
            run.font.size = Pt(9)
            if col_idx == 0:
                run.font.bold = True
            run.font.color.rgb = text_dark

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # 5. Proposed Features to Add / Production Roadmap
    add_heading_1("5. Proposed Production Roadmap: Features for the Main Team")
    add_body(
        "The prototype validates the core recommendation quality. Here is the recommended roadmap of features "
        "the main team can selectively adopt:"
    )

    add_heading_2("Proposal A: PostgreSQL + pgvector Migration (Priority: High)")
    add_body(
        "Migrate from prototype in-memory numpy storage to PostgreSQL with the pgvector extension. "
        "Enables persistent vector storage with HNSW indexing for sub-5ms retrieval across millions of articles."
    )

    add_heading_2("Proposal B: Offline Precomputation Worker (Priority: High)")
    add_body(
        "Deploy a lightweight background task (Celery/RQ) listening to CMS publish webhooks to execute the windowed "
        "similarity search and cross-encoder rerank offline."
    )

    add_heading_2("Proposal C: Real-Time CTR & Multi-Armed Bandit Feedback Loop (Priority: Medium)")
    add_body(
        "Ingest frontend click telemetry. By tracking impression-to-click ratios in real-time, the engine can "
        "dynamically promote high-CTR stories and demote underperforming ones using Thompson Sampling exploration."
    )

    add_heading_2("Proposal D: Maximal Marginal Relevance (MMR) for Diversity (Priority: Medium)")
    add_body(
        "Avoid the 'echo-chamber' trap where 5 recommendations are slight variations of the same breaking story. "
        "An MMR diversification layer balances topical relevance with topic novelty, ensuring readers see a varied mix."
    )

    add_heading_2("Proposal E: Exponential Time Decay for Freshness (Priority: Medium)")
    add_body(
        "Apply a calibrated half-life decay function: Score = Base_Score * exp(-lambda * delta_hours). "
        "Ensures breaking news stays top-of-mind while older stories gracefully deprioritize."
    )

    add_heading_2("Proposal F: Editorial Override & A/B Testing Framework (Priority: Low)")
    add_body(
        "Provide an editorial dashboard where editors can pin 'must-read' stories at Slot #1 while algorithmic "
        "recommendations populate Slots #2 to #5. Includes a 50/50 traffic split router to measure lift in CTR."
    )

    # 6. Integration Architecture & Next Steps
    add_heading_1("6. Integration Architecture & Next Steps")
    add_body(
        "The system connects seamlessly with the existing production stack through three lightweight touchpoints:"
    )
    add_bullet("1. CMS Publishing Webhook: ", "POST /internal/ingest {article_id, title, body, section} -> kicks off background worker.")
    add_bullet("2. FastAPI Recommendation Microservice: ", "GET /recommend?article_id=X&limit=5 -> ultra-fast O(1) response from Postgres/Redis.")
    add_bullet("3. Frontend Analytics Hook: ", "POST /telemetry/click {article_id, clicked_recommendation_id} -> feeds popularity & bandit weights.")

    add_body(
        "Conclusion: By adopting the publish-time execution pipeline and PostgreSQL pgvector, the engineering team "
        "gains state-of-the-art semantic recommendations, zero live server inference overhead, and immediate cold-start support "
        "at a fraction of cloud SaaS costs."
    )

    doc.save(filename)
    print(f"Document successfully created: {filename}")

if __name__ == "__main__":
    create_proposal_document()
