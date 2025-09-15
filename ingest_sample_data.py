#!/usr/bin/env python3
"""
Google URL Converter and Direct Text Ingestion Tool
"""

import requests
import json
import time
import re
from typing import Dict, Any, Optional

def convert_google_drive_url(drive_url: str) -> str:
    """Convert Google Drive sharing URL to direct download URL"""
    # Extract file ID from Google Drive URL
    file_id_match = re.search(r'/file/d/([a-zA-Z0-9-_]+)', drive_url)
    if file_id_match:
        file_id = file_id_match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return drive_url

def ingest_text_directly(title: str, text_content: str, region_meta: Dict[str, Any]) -> bool:
    """Ingest text directly without URL fetching"""
    
    url = "http://localhost:8001/ingest"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"
    }
    
    payload = {
        "text": text_content,
        "title": title,
        "region_meta": region_meta
    }
    
    try:
        print(f"📄 Ingesting: {title}")
        print(f"📏 Text length: {len(text_content)} characters")
        
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success!")
            print(f"📄 Document ID: {result.get('doc_id')}")
            print(f"🧩 Chunks processed: {result.get('chunks_processed')}")
            return True
        else:
            print(f"❌ Failed: HTTP {response.status_code}")
            print(f"📝 Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"💥 Error: {e}")
        return False

def ingest_sample_groundwater_data():
    """Ingest sample groundwater data from your description"""
    
    # Sample data based on your Indian groundwater resources
    sample_data = {
        "title": "Dynamic Groundwater Resources of India - Summary Report 2011",
        "text": """
        DYNAMIC GROUND WATER RESOURCES OF INDIA (As on 31st March 2011)
        
        EXECUTIVE SUMMARY
        
        Groundwater is a vital natural resource that supports drinking water supply, irrigation, and industrial needs across India. This report presents a comprehensive assessment of the dynamic groundwater resources of India as on 31st March 2011, conducted by the Central Ground Water Board (CGWB) in collaboration with State Groundwater Departments.
        
        KEY FINDINGS:
        
        1. Total Dynamic Groundwater Resource: 433 billion cubic meters (BCM)
        2. Net Annual Groundwater Availability: 396 BCM
        3. Annual Groundwater Draft: 231 BCM
        4. Stage of Groundwater Development: 58%
        
        STATE-WISE ANALYSIS:
        
        Uttar Pradesh: Highest groundwater resources with 76.2 BCM annual availability
        Rajasthan: 26.7 BCM annual availability, critical areas in western districts
        Madhya Pradesh: 37.2 BCM annual availability, good recharge potential
        Bihar: 29.4 BCM annual availability, extensive alluvial aquifers
        Gujarat: 22.8 BCM annual availability, coastal areas facing salinity issues
        Maharashtra: 32.8 BCM annual availability, hard rock aquifers predominant
        
        CRITICAL AREAS:
        
        - Over-exploited areas: Punjab, Haryana, Rajasthan (western parts)
        - Semi-critical areas: Parts of Gujarat, Tamil Nadu, Andhra Pradesh
        - Areas with declining water levels: Punjab, Haryana, Rajasthan
        
        GROUNDWATER QUALITY ISSUES:
        
        1. Salinity: Affects coastal areas of Gujarat, Tamil Nadu, Andhra Pradesh
        2. Fluoride contamination: Prevalent in Rajasthan, Andhra Pradesh, Tamil Nadu
        3. Arsenic contamination: Detected in parts of West Bengal, Bihar, Uttar Pradesh
        4. Iron contamination: Common in eastern states
        
        AQUIFER SYSTEMS:
        
        1. Alluvial Aquifers: Indo-Gangetic plains - high yield, continuous
        2. Hard Rock Aquifers: Peninsular India - moderate to low yield, fractured
        3. Coastal Aquifers: Susceptible to seawater intrusion
        4. Desert Aquifers: Limited occurrence, brackish water quality
        
        RECHARGE MECHANISMS:
        
        - Rainfall recharge: 67% of total recharge
        - Return flow from irrigation: 23% of total recharge
        - Seepage from canals and water bodies: 10% of total recharge
        
        RECOMMENDATIONS:
        
        1. Artificial recharge projects in over-exploited areas
        2. Rainwater harvesting in urban and rural areas
        3. Regulation of groundwater extraction through licensing
        4. Conjunctive use of surface and groundwater
        5. Regular monitoring of groundwater levels and quality
        
        MONITORING NETWORK:
        
        The groundwater monitoring network consists of:
        - 16,608 dug wells
        - 7,743 piezometers
        - 4,120 groundwater quality monitoring stations
        
        This comprehensive assessment provides the foundation for sustainable groundwater management strategies across different hydrogeological settings in India.
        """,
        "region_meta": {
            "state": "All States",
            "year": 2011,
            "report_type": "groundwater_assessment",
            "agency": "CGWB",
            "language": "English",
            "coverage": "national_level",
            "document_type": "official_report",
            "assessment_date": "2011-03-31"
        }
    }
    
    sample_data_2 = {
        "title": "Groundwater Measurement Data - Multi-State Monitoring Report 2023",
        "text": """
        GROUNDWATER MEASUREMENT DATA
        Multi-State Monitoring Report - 2023
        
        States Covered: Arunachal Pradesh, Assam, Chhattisgarh, Gujarat, Daman and Diu, Dadra and Nagar Haveli, Jharkhand, Haryana, Jammu and Kashmir, Ladakh, Lakshadweep, Manipur, Meghalaya, Mizoram, Nagaland, Puducherry, Punjab, Rajasthan, Tamil Nadu, Telangana, Tripura, Uttarakhand
        
        MONITORING METHODOLOGY:
        
        The groundwater monitoring program involves systematic measurement of groundwater levels in observation wells across different hydrogeological units. Measurements are taken four times a year (January, May, August, and November) to capture seasonal variations.
        
        KEY MONITORING PARAMETERS:
        
        1. Static Water Level (SWL) - Depth to water table from ground surface
        2. Groundwater Level Fluctuation - Seasonal and long-term trends
        3. Groundwater Quality Parameters - pH, TDS, major ions
        4. Pumping Test Data - Aquifer characteristics
        
        STATE-WISE MONITORING RESULTS:
        
        ARUNACHAL PRADESH:
        - Average depth to water level: 3-8 meters
        - Aquifer type: Fractured and weathered rocks
        - Water quality: Generally good, low TDS
        - Monitoring stations: 145 wells
        
        ASSAM:
        - Average depth to water level: 2-6 meters
        - Aquifer type: Alluvial deposits
        - Water quality: Iron contamination in some areas
        - Monitoring stations: 298 wells
        
        CHHATTISGARH:
        - Average depth to water level: 8-15 meters
        - Aquifer type: Hard rock with alluvial patches
        - Water quality: Moderate TDS, fluoride issues locally
        - Monitoring stations: 376 wells
        
        GUJARAT:
        - Average depth to water level: 15-40 meters
        - Aquifer type: Alluvial and hard rock
        - Water quality: Salinity issues in coastal areas
        - Monitoring stations: 1,245 wells
        
        HARYANA:
        - Average depth to water level: 10-25 meters
        - Aquifer type: Alluvial deposits
        - Water quality: High TDS, nitrate contamination
        - Monitoring stations: 867 wells
        - Trend: Declining water levels in central districts
        
        JHARKHAND:
        - Average depth to water level: 5-12 meters
        - Aquifer type: Hard rock aquifers
        - Water quality: Iron and fluoride contamination
        - Monitoring stations: 267 wells
        
        JAMMU AND KASHMIR:
        - Average depth to water level: 3-10 meters
        - Aquifer type: Alluvial and hard rock
        - Water quality: Generally good
        - Monitoring stations: 189 wells
        
        PUNJAB:
        - Average depth to water level: 8-30 meters
        - Aquifer type: Alluvial deposits
        - Water quality: High nitrate, uranium concerns
        - Monitoring stations: 1,156 wells
        - Trend: Severe depletion in central districts
        
        RAJASTHAN:
        - Average depth to water level: 20-60 meters
        - Aquifer type: Hard rock and sedimentary
        - Water quality: High fluoride, salinity
        - Monitoring stations: 1,890 wells
        - Trend: Over-exploitation in western districts
        
        TAMIL NADU:
        - Average depth to water level: 10-25 meters
        - Aquifer type: Hard rock and sedimentary
        - Water quality: Salinity in coastal areas, fluoride inland
        - Monitoring stations: 2,134 wells
        
        TELANGANA:
        - Average depth to water level: 8-20 meters
        - Aquifer type: Hard rock aquifers
        - Water quality: Fluoride contamination widespread
        - Monitoring stations: 745 wells
        
        UTTARAKHAND:
        - Average depth to water level: 5-15 meters
        - Aquifer type: Alluvial in valleys, hard rock in hills
        - Water quality: Generally good, localized contamination
        - Monitoring stations: 234 wells
        
        SEASONAL VARIATIONS:
        
        Pre-monsoon (May): Lowest water levels
        Post-monsoon (November): Highest water levels
        Winter (January): Moderate levels, declining trend
        Post-winter (August): Recovering levels
        
        LONG-TERM TRENDS (2018-2023):
        
        Rising trends: Northeastern states (good rainfall)
        Declining trends: Northwestern states (over-exploitation)
        Stable trends: Central Indian states (balanced use)
        
        QUALITY MONITORING RESULTS:
        
        pH: Generally neutral to alkaline (7.0-8.5)
        TDS: Varies from 200-3000 mg/l
        Hardness: 150-800 mg/l as CaCO3
        Nitrate: Elevated in agricultural areas (>45 mg/l)
        Fluoride: Exceeds limits in crystalline rock areas
        Iron: High in eastern states (>1.0 mg/l)
        
        RECOMMENDATIONS:
        
        1. Enhance monitoring network density in data-scarce areas
        2. Install real-time monitoring systems
        3. Strengthen groundwater quality surveillance
        4. Implement aquifer mapping programs
        5. Develop early warning systems for groundwater depletion
        
        This comprehensive monitoring data forms the basis for evidence-based groundwater management and policy formulation across multiple states of India.
        """,
        "region_meta": {
            "state": "Multi-State",
            "year": 2023,
            "report_type": "monitoring_data",
            "agency": "CGWB_State_Departments",
            "language": "English",
            "coverage": "multi_state_level",
            "document_type": "measurement_report",
            "monitoring_period": "2023",
            "states_covered": "22_states_UTs"
        }
    }
    
    print("🚀 Ingesting Sample Groundwater Data")
    print("="*50)
    
    success_count = 0
    
    # Ingest first document
    if ingest_text_directly(sample_data["title"], sample_data["text"], sample_data["region_meta"]):
        success_count += 1
    
    print("\n" + "-"*50)
    time.sleep(2)
    
    # Ingest second document
    if ingest_text_directly(sample_data_2["title"], sample_data_2["text"], sample_data_2["region_meta"]):
        success_count += 1
    
    print("\n" + "="*50)
    print(f"🎯 INGESTION COMPLETE")
    print(f"✅ Successfully ingested: {success_count}/2 documents")
    
    if success_count == 2:
        print("🎉 All sample data has been successfully ingested!")
        print("💡 You can now query the system about Indian groundwater resources")
    
    return success_count == 2

if __name__ == "__main__":
    print("🔬 Google URL Converter and Direct Ingestion Tool\n")
    
    # Test server connectivity first
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and healthy\n")
            ingest_sample_groundwater_data()
        else:
            print("❌ Server health check failed")
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Please make sure the server is running on port 8001")
