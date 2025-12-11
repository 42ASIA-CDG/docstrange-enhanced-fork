"""Pure LLM text structuring service for converting OCR text to structured JSON.

This module provides a lightweight text-only LLM service for structuring extracted text
into JSON format. Unlike vision-language models, this uses pure language models which are:
- Faster (no image encoding/processing)
- More efficient (smaller model size)
- Better at text understanding and structuring
- Cheaper to run (less VRAM required)

Supported backends:
- Ollama (local, free, private)
- vLLM (fast, production-ready)
- Transformers (flexible, any HF model)
"""

import logging
import json
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class OllamaTextStructurer:
    """Pure LLM text structuring using local Ollama models.
    
    Best for:
    - Privacy-sensitive documents
    - No cloud API costs
    - Local development
    
    Recommended models:
    - qwen2.5:7b - Excellent JSON structuring
    - llama3.2:3b - Fast and lightweight
    - mistral:7b - Good accuracy
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        """Initialize Ollama text structurer.
        
        Args:
            base_url: Ollama server URL
            model: Model name (qwen2.5:7b recommended for JSON)
        """
        self.base_url = base_url
        self.model = model
        self._client = None
        logger.info(f"Initialized OllamaTextStructurer with model: {model}")
    
    def _get_client(self):
        """Get Ollama client with lazy loading."""
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama"
                )
        return self._client
    
    def structure_text_to_json(self, text: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert extracted text into structured JSON using Ollama LLM.
        
        Args:
            text: Pre-extracted text (from TrOCR, PaddleOCR, etc.)
            json_schema: Optional JSON schema to guide extraction
            
        Returns:
            Structured data as dictionary
        """
        try:
            if json_schema:
                schema_str = json.dumps(json_schema, indent=2)
                prompt = f"""You are a data extraction assistant. Convert the following extracted text into a structured JSON object that matches the provided schema.

Extracted Text:
{text}

Target JSON Schema:
{schema_str}

RULES:
1. Return ONLY valid JSON - no explanations, no markdown, no code blocks
2. Extract values for ALL fields in the schema
3. For missing fields, use null
4. For arrays, return empty array [] if no data found
5. For numeric fields, return numbers not strings
6. Standardize dates to YYYY-MM-DD format
7. Convert Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) to Western (0123456789)
8. Fix obvious OCR errors (l→1, O→0, etc.)
9. For financial docs: verify calculations are correct
10. Remove duplicate entries from arrays

Return the JSON object now:"""
            else:
                prompt = f"""You are a data extraction assistant. Convert the following extracted text into a well-structured JSON object.

Extracted Text:
{text}

RULES:
1. Return ONLY valid JSON - no explanations, no markdown, no code blocks
2. Use descriptive field names based on the content
3. Group related fields logically
4. Use appropriate data types (numbers, strings, arrays, booleans)
5. For repeated items, use arrays
6. Standardize dates to YYYY-MM-DD format
7. Fix obvious OCR errors

Return the JSON object now:"""
            
            client = self._get_client()
            
            logger.info(f"Structuring {len(text)} chars of text with Ollama ({self.model})...")
            print(f"🧠 Structuring text with Ollama ({self.model})...")
            
            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.0,  # Deterministic output
                    "num_predict": 2048,  # Allow larger responses
                    "stop": ["}\n\n", "```"],  # Stop after JSON
                },
                stream=False
            )
            
            response_text = response['response'].strip()
            
            logger.info(f"Ollama generated {len(response_text)} chars")
            print(f"✅ Ollama structuring complete ({len(response_text)} chars)")
            
            # Parse JSON
            parsed = self._parse_json(response_text)
            
            return {
                "structured_data": parsed,
                "format": "ollama_text_to_json",
                "model": self.model,
                "schema": json_schema,
                "source": "text_extraction"
            }
            
        except Exception as e:
            logger.error(f"Ollama text structuring failed: {e}")
            raise
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text, handling markdown and extra text.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON dictionary
        """
        # Remove markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Failed to parse JSON: {text[:500]}")
            return {"error": "Failed to parse JSON", "raw_output": text[:1000]}
    
    def is_available(self) -> bool:
        """Check if Ollama service is available.
        
        Returns:
            True if Ollama is reachable and model is available
        """
        try:
            client = self._get_client()
            client.list()  # Test connection
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False


class vLLMTextStructurer:
    """Pure LLM text structuring using vLLM (fast production deployment).
    
    Best for:
    - Production workloads
    - High throughput requirements
    - GPU inference with batching
    
    Recommended models:
    - Qwen/Qwen2.5-7B-Instruct - Best JSON structuring
    - meta-llama/Llama-3.2-3B-Instruct - Fast and small
    - mistralai/Mistral-7B-Instruct-v0.3 - Good balance
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize vLLM text structurer.
        
        Args:
            model_path: HuggingFace model path
        """
        self.model_path = model_path
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load LLM with vLLM."""
        try:
            from vllm import LLM, SamplingParams
            
            logger.info(f"Loading {self.model_path} with vLLM...")
            print(f"🚀 Loading {self.model_path} with vLLM...")
            
            self.llm = LLM(
                model=self.model_path,
                dtype="half",
                gpu_memory_utilization=0.75,
                max_model_len=8192,
                trust_remote_code=True,
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
            
            logger.info("vLLM text structurer ready")
            print("✅ vLLM text structurer loaded")
            
        except ImportError:
            raise ImportError("vLLM required. Install with: pip install vllm>=0.6.0")
        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise
    
    def structure_text_to_json(self, text: str, json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert extracted text into structured JSON using vLLM.
        
        Args:
            text: Pre-extracted text
            json_schema: Optional JSON schema
            
        Returns:
            Structured data dictionary
        """
        try:
            if json_schema:
                schema_str = json.dumps(json_schema, indent=2)
                system_msg = "You are a data extraction assistant that converts text to JSON."
                user_msg = f"""Convert this extracted text into structured JSON matching the schema.

Extracted Text:
{text}

Target Schema:
{schema_str}

Return ONLY the JSON object - no explanations."""
            else:
                system_msg = "You are a data extraction assistant that converts text to JSON."
                user_msg = f"""Convert this extracted text into a well-structured JSON object.

Extracted Text:
{text}

Return ONLY the JSON object - no explanations."""
            
            # Construct prompt (Qwen2.5 format)
            prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_msg}<|im_end|>
<|im_start|>assistant
"""
            
            logger.info(f"Structuring {len(text)} chars with vLLM...")
            print(f"⚡ Structuring text with vLLM (fast)...")
            
            outputs = self.llm.generate(prompt, sampling_params=self.sampling_params)
            response_text = outputs[0].outputs[0].text.strip()
            
            logger.info(f"vLLM generated {len(response_text)} chars")
            print(f"✅ vLLM structuring complete")
            
            # Parse JSON
            parsed = self._parse_json(response_text)
            
            return {
                "structured_data": parsed,
                "format": "vllm_text_to_json",
                "model": self.model_path,
                "schema": json_schema,
                "source": "text_extraction"
            }
            
        except Exception as e:
            logger.error(f"vLLM text structuring failed: {e}")
            raise
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from text."""
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            return {"error": "Failed to parse JSON", "raw_output": text[:1000]}


def create_text_structurer(backend: str = "ollama", **kwargs):
    """Factory function to create text structurer based on backend.
    
    Args:
        backend: "ollama" or "vllm"
        **kwargs: Backend-specific arguments
        
    Returns:
        Text structurer instance
    """
    if backend == "ollama":
        return OllamaTextStructurer(**kwargs)
    elif backend == "vllm":
        return vLLMTextStructurer(**kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'ollama' or 'vllm'")
