from __future__ import annotations

BENCHMARK_CASES = [
    {
        "id": "licence_pay_and_recover",
        "description": "Fatal commercial truck accident where the insurer raises an unlicensed-driver defense.",
        "prompt": (
            "Client: Mrs. Lakshmi Devi\n"
            "Matter: Motor accident claim - death of spouse\n\n"
            "Mrs. Lakshmi Devi's husband was killed in a road accident involving a commercial truck. "
            "The truck driver was operating the vehicle without a valid driving license at the time of the accident. "
            "The insurance company is denying liability and says the policy is void because the driver was unlicensed.\n\n"
            "Please provide supporting precedents, adverse precedents, and a strategy recommendation."
        ),
        "gold_support_docs": ["doc_032.pdf", "doc_006.pdf", "doc_025.pdf", "doc_029.pdf"],
        "gold_adverse_docs": ["doc_031.pdf", "doc_034.pdf", "doc_028.pdf", "doc_033.pdf"],
        "gold_relevant_docs": [
            "doc_006.pdf",
            "doc_025.pdf",
            "doc_028.pdf",
            "doc_029.pdf",
            "doc_031.pdf",
            "doc_032.pdf",
            "doc_033.pdf",
            "doc_034.pdf",
        ],
        "retrieval_must_cover_docs": ["doc_032.pdf", "doc_029.pdf", "doc_031.pdf", "doc_034.pdf"],
        "support_reasoning_terms": ["third party", "pay and recover", "compensation", "liable"],
        "adverse_reasoning_terms": ["breach", "not liable", "policy", "entrusted"],
        "reasoning_fact_terms": ["commercial truck", "unlicensed", "driving license", "insurer"],
        "expected_support_signal_terms": ["supports", "liable", "pay and recover", "compensation"],
        "expected_adverse_signal_terms": ["risk", "breach", "not liable", "defence", "policy"],
    },
    {
        "id": "contributory_negligence_truck",
        "description": "Truck-accident prompt focused on contributory negligence and claimant-versus-insurer alignment.",
        "prompt": (
            "Find precedents in this corpus on contributory negligence in truck accident claims. "
            "Tell me which cases help the claimant and which help the insurer."
        ),
        "gold_support_docs": ["doc_018.pdf", "doc_023.pdf", "doc_009.pdf"],
        "gold_adverse_docs": ["doc_028.pdf", "doc_026.pdf", "doc_035.pdf"],
        "gold_relevant_docs": ["doc_009.pdf", "doc_018.pdf", "doc_023.pdf", "doc_026.pdf", "doc_028.pdf", "doc_035.pdf"],
        "retrieval_must_cover_docs": ["doc_018.pdf", "doc_023.pdf", "doc_028.pdf", "doc_026.pdf"],
        "support_reasoning_terms": ["contributory", "negligence", "claimant", "truck"],
        "adverse_reasoning_terms": ["insurer", "breach", "policy", "liable"],
        "reasoning_fact_terms": ["contributory negligence", "truck", "claimant", "insurer"],
        "expected_support_signal_terms": ["supports", "claimant", "contributory", "negligence"],
        "expected_adverse_signal_terms": ["risk", "insurer", "breach", "liable"],
    },
    {
        "id": "commercial_vehicle_liability",
        "description": "Broad commercial-vehicle precedent search that should surface both useful and risky transport cases.",
        "prompt": (
            "Find precedents that discuss commercial vehicles, goods carriages, or transport-company trucks "
            "in motor accident claims, and separate the useful ones from the risky ones."
        ),
        "gold_support_docs": ["doc_027.pdf", "doc_029.pdf", "doc_032.pdf"],
        "gold_adverse_docs": ["doc_031.pdf", "doc_034.pdf"],
        "gold_relevant_docs": ["doc_027.pdf", "doc_029.pdf", "doc_031.pdf", "doc_032.pdf", "doc_034.pdf"],
        "retrieval_must_cover_docs": ["doc_027.pdf", "doc_029.pdf", "doc_031.pdf"],
        "support_reasoning_terms": ["commercial", "goods carriage", "truck", "motor accident"],
        "adverse_reasoning_terms": ["breach", "insurer", "not liable", "policy"],
        "reasoning_fact_terms": ["commercial vehicle", "goods carriage", "truck", "transport"],
        "expected_support_signal_terms": ["supports", "commercial", "goods carriage", "truck"],
        "expected_adverse_signal_terms": ["risk", "breach", "insurer", "policy"],
    },
]
