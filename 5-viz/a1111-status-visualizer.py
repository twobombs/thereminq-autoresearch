import os
import base64
import requests
from datetime import datetime

# ==============================================================================
# Configuration & Endpoints
# ==============================================================================

# Local LLM for Prompt Translation
LLM_API_URL = os.getenv("LLM_API_URL", "http://127.0.0.1:8080/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "nemotron-orchestrator-8b")

# Automatic1111 API Endpoint
A1111_URL = os.getenv("A1111_URL", "http://127.0.0.1:7860/sdapi/v1/txt2img")

# Output directory for generated status dashboards
OUTPUT_DIR = "wiki/assets/status_snapshots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# Phase 1: AI-Driven Prompt Formulation
# ==============================================================================

def formulate_visual_prompt(orchestrator_data: str) -> str:
    """
    Takes the raw analytical output from the Orchestrator and asks the local LLM 
    to translate it into a highly descriptive visual prompt for Stable Diffusion.
    """
    print("\n[1] 🧠 PROMPT TRANSLATION: Formulating visual directives from data...")
    
    system_prompt = """You are a highly skilled Prompt Engineer for Stable Diffusion.
Your job is to read the raw Agile project state/orchestrator data and translate it into a rich, highly descriptive image generation prompt.
We want to generate a "Holoqubed" 3D Sci-Fi UI Dashboard or abstract visual metaphor that represents the current state of the project.
If the project is blocked, make it look warning-red/glitchy. If it's active and healthy, make it glowing cyan/green and highly structured.

Output ONLY the comma-separated keywords for the Stable Diffusion prompt. No conversational text.
Include structural tags like: masterpiece, best quality, highly detailed, isometric 3d, glowing holograms, UI/UX dashboard."""

    try:
        response = requests.post(
            LLM_API_URL,
            headers={"Content-Type": "application/json", "Authorization": "Bearer local-sk"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Translate this project data into a visual prompt:\n\n{orchestrator_data}"}
                ],
                "temperature": 0.6,
                "max_tokens": 150
            },
            timeout=30
        )
        response.raise_for_status()
        sd_prompt = response.json()['choices'][0]['message']['content'].strip()
        
        # Clean up in case the LLM wrapped it in quotes
        if sd_prompt.startswith('"') and sd_prompt.endswith('"'):
             sd_prompt = sd_prompt[1:-1]
             
        print(f"    [+] Generated A1111 Prompt: '{sd_prompt}'")
        return sd_prompt

    except Exception as e:
        print(f"    [!] Failed to generate prompt: {e}")
        # Fallback prompt
        return "masterpiece, highly detailed, isometric 3d, glowing holographic ui dashboard, floating data nodes, abstract technology, cyan and blue tones, dark background"

# ==============================================================================
# Phase 2: Automatic1111 Generation
# ==============================================================================

def generate_status_image(sd_prompt: str) -> str:
    """
    Sends the formulated prompt to the local Automatic1111 API and saves the image.
    """
    print("\n[2] 🎨 A1111 GENERATION: Sending directives to Automatic1111...")
    
    payload = {
        "prompt": sd_prompt,
        "negative_prompt": "worst quality, low quality, messy, text, watermark, signature, ugly, blurry, deformed",
        "steps": 25,
        "sampler_name": "DPM++ 2M Karras",
        "cfg_scale": 7.0,
        "width": 1024,
        "height": 576, # 16:9 cinematic aspect ratio for a dashboard
        "restore_faces": False,
        "do_not_save_samples": False,
        "do_not_save_grid": False
    }

    try:
        response = requests.post(A1111_URL, json=payload, timeout=120) # A1111 might take a minute
        response.raise_for_status()
        
        r_json = response.json()
        
        # A1111 returns images as a list of base64 strings
        base64_image = r_json['images'][0]
        
        # Save the file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Project_Snapshot_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, "wb") as fh:
            fh.write(base64.b64decode(base64_image))
            
        print(f"    [+] Success! Visual output saved to: {filepath}")
        return filepath

    except requests.exceptions.ConnectionError:
         print("    [!] Error: Could not connect to A1111. Is it running with --api flag on port 7860?")
    except Exception as e:
        print(f"    [!] Error during A1111 generation: {e}")
        
    return None

# ==============================================================================
# Pipeline Execution
# ==============================================================================

if __name__ == "__main__":
    # In a full pipeline, this data would be passed directly from Orchestrator/AgenticAgile.py
    sample_orchestrator_data = """
    DAILY SYNTHESIS:
    - 3 Active Tasks (API upgrades, Frontend React Components)
    - 1 Blocked Task (Missing database credentials)
    - Overall Velocity: High
    - Risk Level: Medium (due to blocker)
    """
    
    print("=== STARTING INTERMEDIARY VISUAL OUTPUT GENERATOR ===")
    
    # Step 1: LLM translates data to Image Prompt
    stable_diffusion_prompt = formulate_visual_prompt(sample_orchestrator_data)
    
    # Step 2: A1111 generates the UI/Visual Metaphor
    image_path = generate_status_image(stable_diffusion_prompt)
    
    if image_path:
        print("\n=== PIPELINE COMPLETE ===")
    else:
        print("\n=== PIPELINE FAILED ===")
