import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from .cluster_aspect_extractor import ClusterAspectExtractor

# Load environment variables (API Key)
load_dotenv()

class InsightEngine:
    def __init__(self):
        self.cluster_extractor = ClusterAspectExtractor()
        # Initialize OpenAI Client pointing to OpenRouter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in .env file")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/mukesh-5059/product-review-api", # Optional
                "X-Title": "Product Review Insights API", # Optional
            }
        )
        # Using a reliable, high-quality, and fast model via OpenRouter
        self.model_id = "openai/gpt-4o-mini"

    def get_top_aspects(self, product_id: str):
        """
        1. Clusters reviews to get representative sentences.
        2. Sends them to OpenRouter to extract clean, distinct aspects.
        3. Scales the number of aspects based on the amount of data.
        """
        # Step 1: Get representative sentences (max 25)
        representative_sentences = self.cluster_extractor.get_representative_sentences(product_id)

        if not representative_sentences:
            return {"error": "Not enough data for this product."}

        num_reps = len(representative_sentences)

        # Step 2: Determine target number of aspects based on data volume
        if num_reps <= 5:
            target_range = "2 to 4"
        elif num_reps <= 12:
            target_range = "5 to 7"
        elif num_reps <= 20:
            target_range = "7 to 9"
        else:
            target_range = "8 to 10"

        # Step 3: Prepare Prompt
        sentences_text = "\n".join([f"- {s}" for s in representative_sentences])

        prompt = f"""
        You are an expert product analyst. I will provide you with a set of representative sentences from customer reviews for a specific product.
        Your task is to identify the top {target_range} distinct 'Aspects' or 'Topics' that customers care about.

        Mandatory Guidelines:
        1. UNIVERSAL MANDATE: You MUST include 'Price/Value', 'Delivery/Packaging', and 'Customer Service' if there is ANY mention of them in the provided sentences. For large datasets, these are essential benchmarks.
        2. CRITICISM FOCUS: Analyze both praise and common complaints. If users criticize a specific flaw, capture it as a distinct aspect (e.g., 'Aftertaste', 'Build Quality').
        3. PRODUCT FEATURES: Identify the core unique features that distinguish this specific product.
        4. DIVERSITY: Ensure the {target_range} aspects cover the full spectrum of the customer experience provided in the sentences.
        5. FORMAT: Return ONLY a valid JSON list of strings.

        Representative Sentences:
        {sentences_text}

        JSON Output:
        """

        # Step 4: Call OpenRouter
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are an assistant that returns only structured JSON output."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" } # Ensures OpenAI-style JSON mode
            )
            
            content = response.choices[0].message.content
            
            # OpenRouter models sometimes wrap JSON in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            # If the LLM returned a dict with a key like {"aspects": [...]}, extract the list
            if isinstance(data, dict):
                # Try common keys or just take the first list found
                if "aspects" in data: return data["aspects"]
                if "topics" in data: return data["topics"]
                for val in data.values():
                    if isinstance(val, list): return val
            
            return data
        except Exception as e:
            return {"error": f"OpenRouter Request Failed: {str(e)}"}

if __name__ == "__main__":
    import sys
    
    try:
        engine = InsightEngine()
        test_id = sys.argv[1] if len(sys.argv) > 1 else "B003VXFK44"
        
        print(f"\n🔍 Extracting Top Aspects via OpenRouter for Product: {test_id}...")
        aspects = engine.get_top_aspects(test_id)
        
        if isinstance(aspects, list):
            print(f"✅ Found {len(aspects)} Aspects:")
            for i, aspect in enumerate(aspects):
                print(f"  {i+1}. {aspect}")
        else:
            print(f"❌ Error: {aspects.get('error') if isinstance(aspects, dict) else aspects}")
    except Exception as e:
        print(f"❌ Initialization Error: {e}")
