"""Quick usage examples for the RFP summarizer."""

import json
from rfp_summarize import summarize_rfp

# --- Example 1: Summarize from a plain-text RFP snippet ---
SAMPLE_RFP = """
REQUEST FOR PROPOSAL
Solicitation No.: ABC-2025-001
Issued by: Department of Public Works
Issue Date: January 15, 2025

PROJECT: City Hall Network Infrastructure Upgrade

INTRODUCTION
The Department of Public Works (DPW) is soliciting proposals from qualified vendors
to design, supply, install, and configure a modern network infrastructure for City Hall.

SCOPE OF WORK
- Replace all network switches and routers (approx. 120 devices)
- Install and configure a next-generation firewall
- Deploy Wi-Fi 6E access points throughout the building (approx. 80 APs)
- Provide network monitoring and management software
- Deliver training for IT staff (minimum 16 hours)

MANDATORY REQUIREMENTS
- Vendor must hold a current Cisco or Juniper Gold Partner certification
- All hardware must carry a minimum 5-year warranty
- Proposal must include a detailed project timeline

PREFERRED QUALIFICATIONS
- Experience with municipal government clients
- ISO 27001 certification

EVALUATION CRITERIA
1. Technical approach and solution quality   – 40%
2. Price / total cost of ownership           – 30%
3. Vendor qualifications and past performance – 20%
4. Project timeline and implementation plan  – 10%

BUDGET
Estimated budget: $750,000 (fixed-price contract)

CONTRACT PERIOD
12 months from Notice to Proceed, with two 1-year optional extensions.

SUBMISSION DEADLINE
February 28, 2025 at 5:00 PM Eastern Time

Questions must be submitted by February 1, 2025.

SUBMISSION INSTRUCTIONS
Submit one (1) original and two (2) hard copies plus one electronic copy (USB) to:
  Procurement Office, City Hall, Room 210

CONTACT
Jane Smith, Procurement Officer
Email: jane.smith@city.gov
Phone: (555) 123-4567
"""

if __name__ == "__main__":
    print("Running RFP summarizer on sample text...\n")
    summary = summarize_rfp(SAMPLE_RFP, output_format="both")
    print("\n\nReturned dict keys:", list(summary.keys()))
