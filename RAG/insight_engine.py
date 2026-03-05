import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from .cluster_aspect_extractor import ClusterAspectExtractor
from .vector_store import VectorStore

# Load environment variables (API Key)
load_dotenv()

class InsightEngine:
    def __init__(self):
        self.cluster_extractor = ClusterAspectExtractor()
        self.vs = VectorStore()
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_id = "openai/gpt-4o-mini"

    def get_full_insights(self, product_id: str):
        """
        Full Task 5 Pipeline with 'Insufficient Data' handling.
        """
        results_raw = self.vs.get_all_for_product(product_id)
        sentences = results_raw.get('documents', [])
        
        if len(sentences) < 5:
            return {
                "product_id": product_id,
                "error": "Not enough data to generate reliable insights. Minimum 5 review sentences required.",
                "confidence": 0.0
            }

        aspects = self.get_top_aspects(product_id)
        if isinstance(aspects, dict) and "error" in aspects:
            return aspects

        results = {
            "product_id": product_id,
            "top_aspects": [],
            "summary": "",
            "confidence": 0.85
        }

        all_evidence_for_summary = []

        for aspect in aspects:
            search_results = self.vs.search(aspect, product_id=product_id, top_k=50)
            
            docs = search_results.get('documents', [[]])[0]
            metas = search_results.get('metadatas', [[]])[0]
            distances = search_results.get('distances', [[]])[0]
            
            pros_evidence = []
            cons_evidence = []
            neutral_evidence = []
            
            RELEVANCE_THRESHOLD = 0.75 
            
            for i in range(len(docs)):
                if distances[i] > RELEVANCE_THRESHOLD:
                    continue
                
                text = docs[i].strip()
                rating = metas[i].get('rating', 0)
                
                # Check for duplicates
                if any(e == text for e in pros_evidence + cons_evidence + neutral_evidence):
                    continue

                if rating >= 4:
                    pros_evidence.append(text)
                elif rating <= 2:
                    cons_evidence.append(text)
                else:
                    neutral_evidence.append(text)

            # --- INSIGHTS LOGIC ---
            total_pro_con = len(pros_evidence) + len(cons_evidence)
            
            # MANDATE: If total sentiment evidence is too low (< 3 sentences), mark as Insufficient
            if total_pro_con < 3:
                category = "Insufficient Data"
                sentiment_score = 0.0
            else:
                sentiment_score = len(pros_evidence) / total_pro_con
                if sentiment_score >= 0.7: category = "Pro"
                elif sentiment_score <= 0.3: category = "Con"
                else: category = "Mixed"

            # Step 4: Append Aspect Data
            aspect_data = {
                "aspect": aspect,
                "category": category,
                "sentiment_score": round(sentiment_score, 2),
                "pros_evidence": pros_evidence[:5],
                "cons_evidence": cons_evidence[:5],
                "reference_evidence": neutral_evidence[:3] if category == "Insufficient Data" else []
            }
            
            results["top_aspects"].append(aspect_data)
            all_evidence_for_summary.extend(pros_evidence[:2] + cons_evidence[:2])

        # Final Summary
        if all_evidence_for_summary:
            summary_text = "\n".join(all_evidence_for_summary[:15])
            prompt = f"Based on these specific review points, provide a concise 2-sentence summary of this product's strengths and weaknesses.\n\nPoints:\n{summary_text}"
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}]
                )
                results["summary"] = response.choices[0].message.content.strip()
            except:
                results["summary"] = "Summary unavailable."

        return results

    def get_top_aspects(self, product_id: str):
        representative_sentences = self.cluster_extractor.get_representative_sentences(product_id)
        if not representative_sentences:
            return {"error": "Not enough data for this product."}

        num_reps = len(representative_sentences)
        if num_reps <= 8: specific_count = "2 to 3"
        elif num_reps <= 15: specific_count = "3 to 4"
        else: specific_count = "5 to 6"

        sentences_text = "\n".join([f"- {s}" for s in representative_sentences])
        prompt = f"""
        Identify aspects for this product.
        Include Mandatory: 'Price/Value', 'Delivery/Packaging', and 'Customer Service'.
        Identify {specific_count} additional distinct product-specific aspects.
        NO REDUNDANCY. NO GENERIC SENTIMENT.
        Return ONLY a JSON list of strings.
        Reviews:
        {sentences_text}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "system", "content": "Return JSON list only."}, {"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Robust parsing for list or dict
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list): return val
                return list(data.keys())
            return data
        except Exception as e:
            return {"error": f"Failed to extract aspects: {str(e)}"}

if __name__ == "__main__":
    import sys
    engine = InsightEngine()
    
    # Testing ONLY low/medium data products as requested
    test_ids = ["B002BCD2OG", "B004NE2E9O"]
    
    for test_id in test_ids:
        print("\n" + "="*80)
        print(f"🔍 ANALYZING PRODUCT (LOW DATA TEST): {test_id}")
        print("="*80)
        
        insights = engine.get_full_insights(test_id)
        
        if "error" in insights and "top_aspects" not in insights:
            print(f"❌ Error: {insights['error']}")
            continue

        print(f"SUMMARY: {insights.get('summary')}\n")
        
        for aspect in insights.get('top_aspects', []):
            if aspect['category'] == "Insufficient Data":
                color, label = "⚠️", "INSUFFICIENT DATA"
            else:
                color = "✅" if aspect['category'] == "Pro" else "❌" if aspect['category'] == "Con" else "⚖️"
                label = aspect['category']

            print(f"{color} ASPECT: {aspect['aspect']} ({label})")
            
            if aspect['category'] == "Insufficient Data":
                print("  RETRIEVED REFERENCE REVIEWS (Not enough for Pro/Con categorization):")
                combined = aspect['pros_evidence'] + aspect['cons_evidence'] + aspect['reference_evidence']
                for e in combined[:5]:
                    print(f"    ? {e}")
            else:
                if aspect['pros_evidence']:
                    print("  PRO EVIDENCE:")
                    for e in aspect['pros_evidence']: print(f"    + {e}")
                if aspect['cons_evidence']:
                    print("  CON EVIDENCE:")
                    for e in aspect['cons_evidence']: print(f"    - {e}")
            print("-" * 40)
