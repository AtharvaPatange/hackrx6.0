
#!/usr/bin/env python3
"""
Test script to process the provided questions and show detailed output
"""

import asyncio
import time
import json
from main import app
from fastapi.testclient import TestClient
from schemas import RunRequest

# Test data
test_data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# Expected answers for comparison
expected_answers = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]

def test_questions():
    """Test the questions processing with detailed output"""
    print("=" * 80)
    print("HACKRX QUESTION PROCESSING TEST")
    print("=" * 80)
    
    # Create test client
    client = TestClient(app)
    
    # Prepare request
    request_data = RunRequest(**test_data)
    
    print(f"\n📄 Document URL: {request_data.documents}")
    print(f"❓ Number of questions: {len(request_data.questions)}")
    
    print("\n" + "=" * 50)
    print("PROCESSING QUESTIONS")
    print("=" * 50)
    
    # Print all questions
    for i, question in enumerate(request_data.questions, 1):
        print(f"{i:2d}. {question}")
    
    print("\n" + "=" * 50)
    print("STARTING API CALL")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Make the API call
        response = client.post(
            "/hackrx/run",
            json=request_data.model_dump(),
            headers={"Authorization": "Bearer b3c00e5d9170676e30277fe0ad6d201ffdfd529c4ddb882ad71bf406454178f3"}
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n⏱️  Total processing time: {processing_time:.2f} seconds")
        
        if response.status_code == 200:
            result = response.json()
            answers = result.get('answers', [])
            
            print("\n" + "=" * 50)
            print("ANSWERS RECEIVED")
            print("=" * 50)
            
            for i, (question, answer, expected) in enumerate(zip(request_data.questions, answers, expected_answers), 1):
                print(f"\n❓ Question {i}:")
                print(f"   {question}")
                print(f"\n💡 Answer {i}:")
                print(f"   {answer}")
                print(f"\n🎯 Expected:")
                print(f"   {expected}")
                print("-" * 80)
                
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_questions() 