# backend/mock_retriever.py
from typing import List, Dict

MOCK_CORPUS = [
    {
        "id": "10423",
        "title": "Statin therapy and risk of vascular events",
        "text": "Statins significantly reduce the risk of major vascular events including ischemic stroke and myocardial infarction across diverse patient populations.",
        "score": 0.89
    },
    {
        "id": "20581",
        "title": "MicroRNA profiling in oncology",
        "text": "Altered expression of microRNA clusters regulates oncogenes and tumor suppressors, indicating strong diagnostic potential in early-stage solid tumors.",
        "score": 0.82
    },
    {
        "id": "30114",
        "title": "Adverse drug reactions and long-term lipid-lowering",
        "text": "Statin treatment demonstrates a favorable safety profile; common adverse effects include mild myopathy, with severe hepatic complications remaining exceptionally rare.",
        "score": 0.74
    }
]

def mock_retrieve(query: str, method: str = "semantic", top_k: int = 3) -> List[Dict]:
    """
    Simulates retrieval output.
    When Laiba finishes, replace this with:
    from retrieval import search_scifact
    return search_scifact(query, method, top_k)
    """
    return MOCK_CORPUS[:top_k]