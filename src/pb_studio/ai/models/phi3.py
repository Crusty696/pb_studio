"""
Phi-3 Mini ONNX Wrapper
-----------------------
Implementation of Phi-3 Mini (Logic/Storytelling) using ONNX Runtime GenAI.
Leverages DirectML for hardware acceleration on Windows.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

# Import PB Studio Config/Utils
from ...utils.logger import get_logger

logger = get_logger(__name__)

class Phi3ONNX:
    """
    Wrapper for Phi-3 Mini 4-bit ONNX using onnxruntime-genai.
    """

    def __init__(self, model_path: Path, hardware_strategy: str = "cpu"):
        self.model_path = model_path
        self.strategy = hardware_strategy
        self.model = None
        self.tokenizer = None
        self.tokenizer_stream = None
        
        self._load_model()

    def _find_model_dir(self, base_path: Path) -> Path:
        """Finds the directory containing genai_config.json, preferring DirectML."""
        # 1. DirectML Strategy: Look for specific DML folder first
        if self.strategy == "directml":
            # Common HF patterns for DML optimized models
            for subdir in ["directml", "directml-int4-awq-block-128"]:
                 candidates = list(base_path.rglob(subdir))
                 for c in candidates:
                     if (c / "genai_config.json").exists():
                         return c
        
        # 2. General Recursion (fallback)
        if (base_path / "genai_config.json").exists():
            return base_path
            
        for path in base_path.rglob("genai_config.json"):
            # Avoid picking CPU model if we want DML (simple heuristic: path doesn't contain 'cpu')
            if self.strategy == "directml" and "cpu" in str(path).lower():
                continue
            return path.parent
            
        return base_path

    def _load_model(self):
        """
        Loads the model using onnxruntime-genai.
        """
        try:
            import onnxruntime_genai as og
            
            real_model_path = self._find_model_dir(self.model_path)
            logger.info(f"Loading Phi-3 Mini from {real_model_path} ({self.strategy})...")
            
            # Initialize Model
            self.model = og.Model(str(real_model_path))
            self.tokenizer = og.Tokenizer(self.model)
            
            # Setup Stream
            try:
                self.tokenizer_stream = self.tokenizer.create_stream()
            except AttributeError:
                 # Fallback for different API versions
                 self.tokenizer_stream = og.TokenizerStream(self.tokenizer)
            
            logger.info("✅ Phi-3 Mini loaded successfully.")
            
        except ImportError:
            logger.error("onnxruntime_genai not installed. Phi-3 will not run.")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Phi-3: {e}")
            self.model = None

    def generate_story(self, vision_descriptions: list[str]) -> str:
        """
        Generates a coherent story structure from a list of vision descriptions.
        """
        if not self.model or not self.tokenizer:
            return "Error: Model not loaded."

        import onnxruntime_genai as og

        prompt = self._construct_prompt(vision_descriptions)
        
        try:
            logger.info("Running Phi-3 Inference...")
            
            # Search Options
            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=1024)
            # params.input_ids = ...  <-- Removed in 0.10.0?
            
            generator = og.Generator(self.model, params)
            
            # Pass input tokens to generator
            input_ids = self.tokenizer.encode(prompt)
            generator.append_tokens(input_ids)
            
            full_response = ""
            
            while not generator.is_done():
                generator.generate_next_token()
                
                new_token = generator.get_next_tokens()[0]
                text = self.tokenizer_stream.decode(new_token)
                full_response += text
                # print(text, end='', flush=True) # Optional real-time logging

            # Cleanup
            del generator
            
            return full_response.strip()

        except Exception as e:
            logger.error(f"Phi-3 Generation failed: {e}")
            return f"Error generating story: {e}"

    def _construct_prompt(self, descriptions: list[str]) -> str:
        """Builds the Phi-3 prompt."""
        context = "\n".join([f"- Clip {i+1}: {desc}" for i, desc in enumerate(descriptions)])
        return f"<|user|>\nCreate a short video story plan based on these clips. Assign a role (Intro/Action/Calm) to each:\n{context}\n<|end|>\n<|assistant|>"
