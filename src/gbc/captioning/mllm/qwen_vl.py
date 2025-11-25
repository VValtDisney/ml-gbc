"""
Qwen-VL MLLM Query Module for GBC
==================================
Implements GBC captioning queries using Qwen2-VL vision-language model.

This module provides:
- Image-level scene description
- Entity detection and description
- Spatial relationship extraction
- Composition analysis
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class QwenVLBase:
    """Base class for Qwen-VL queries."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-7B-Instruct", **kwargs):
        """
        Initialize Qwen-VL model.
        
        Args:
            model_name: HuggingFace model identifier
            **kwargs: Additional model configuration
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._initialized = False
        
        logger.info(f"QwenVL initialized with model: {model_name}")
        
    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self.model is None:
            try:
                from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
                import torch
                
                logger.info(f"Loading Qwen2-VL model: {self.model_name}")
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"  # Automatic device placement
                )
                
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self._initialized = True
                
                logger.info("Qwen2-VL model loaded successfully")
                
            except ImportError as e:
                logger.error(f"Failed to import Qwen2-VL: {e}")
                logger.error("Install with: pip install qwen-vl-utils accelerate")
                raise
            except Exception as e:
                logger.error(f"Failed to load Qwen2-VL model: {e}")
                raise
                
    def query(self, image_path: str, prompt: str, max_tokens: int = 512) -> str:
        """
        Query Qwen-VL with image and text prompt.
        
        Args:
            image_path: Path to image file
            prompt: Text prompt for the model
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response as string
        """
        self._ensure_model()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate response
        import torch
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=False  # Deterministic for consistency
            )
        
        # Decode response
        response = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response


class QwenVLImageQuery(QwenVLBase):
    """
    Image-level query for overall scene description.
    
    Generates high-level scene description and identifies
    major entities suitable for detection.
    """
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        system_file: Optional[str] = None,
        query_file: Optional[str] = None,
        suitable_for_detection_func: Optional[callable] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        # Load prompts from files if provided
        self.system_prompt = self._load_prompt(system_file) if system_file else ""
        self.query_prompt = self._load_prompt(query_file) if query_file else self._default_image_prompt()
        self.suitable_for_detection_func = suitable_for_detection_func
        
    def _load_prompt(self, file_path: str) -> str:
        """Load prompt from file."""
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt from {file_path}: {e}")
            return ""
            
    def _default_image_prompt(self) -> str:
        """Default prompt for image-level description."""
        return """Describe this image in detail. Focus on:
1. The overall scene and setting
2. Major objects and entities present
3. Spatial layout and composition
4. Notable features or characteristics

Provide a structured description that identifies distinct entities."""
        
    def __call__(self, image_path: str) -> Dict[str, Any]:
        """
        Generate image-level description.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with scene description and detected entities
        """
        # Combine system and query prompts
        full_prompt = f"{self.system_prompt}\n\n{self.query_prompt}" if self.system_prompt else self.query_prompt
        
        # Query model
        response = self.query(image_path, full_prompt)
        
        # Parse response (simplified - GBC has more sophisticated parsing)
        return {
            "scene_description": response,
            "entities": self._extract_entities(response),
            "raw_response": response
        }
        
    def _extract_entities(self, description: str) -> List[str]:
        """
        Extract entity names from description.
        
        This is a simplified version. GBC has more sophisticated
        entity extraction logic.
        """
        # Simple keyword extraction (placeholder)
        # In production, use NLP or structured output
        entities = []
        
        # Look for common entity patterns
        import re
        # Match capitalized words (potential entities)
        potential_entities = re.findall(r'\b[A-Z][a-z]+\b', description)
        
        # Filter using suitable_for_detection if available
        if self.suitable_for_detection_func:
            entities = [e for e in potential_entities if self.suitable_for_detection_func(e)]
        else:
            entities = potential_entities[:10]  # Limit to top 10
            
        return list(set(entities))  # Remove duplicates


class QwenVLEntityQuery(QwenVLBase):
    """
    Entity-level query for detailed entity description.
    
    Given an entity name and image region, provides detailed
    description of that specific entity.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        system_file: Optional[str] = None,
        query_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        self.system_prompt = self._load_prompt(system_file) if system_file else ""
        self.query_prompt = self._load_prompt(query_file) if query_file else self._default_entity_prompt()
        
    def _load_prompt(self, file_path: str) -> str:
        """Load prompt from file."""
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt from {file_path}: {e}")
            return ""
            
    def _default_entity_prompt(self) -> str:
        """Default prompt for entity description."""
        return """Describe the {entity_name} in this image. Provide:
1. Detailed visual description
2. Color, texture, and appearance
3. Size and scale relative to other objects
4. Any distinctive features

Be specific and detailed."""
        
    def __call__(self, image_path: str, entity_name: str, bbox: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Generate entity-level description.
        
        Args:
            image_path: Path to image file
            entity_name: Name of entity to describe
            bbox: Optional bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with entity description
        """
        # Format prompt with entity name
        prompt = self.query_prompt.format(entity_name=entity_name)
        
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"
            
        # Query model
        response = self.query(image_path, prompt, max_tokens=256)
        
        return {
            "entity_name": entity_name,
            "description": response,
            "bbox": bbox,
            "raw_response": response
        }


class QwenVLRelationQuery(QwenVLBase):
    """
    Relation query for spatial relationships between entities.
    
    Identifies how entities relate to each other spatially.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        system_file: Optional[str] = None,
        query_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        self.system_prompt = self._load_prompt(system_file) if system_file else ""
        self.query_prompt = self._load_prompt(query_file) if query_file else self._default_relation_prompt()
        
    def _load_prompt(self, file_path: str) -> str:
        """Load prompt from file."""
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt from {file_path}: {e}")
            return ""
            
    def _default_relation_prompt(self) -> str:
        """Default prompt for relation extraction."""
        return """Describe the spatial relationships between objects in this image.
For each pair of objects, specify:
- Which object is on the left/right
- Which object is above/below
- Which object is in front/behind
- Any containment relationships (inside, on top of, etc.)

Be precise and systematic."""
        
    def __call__(self, image_path: str, entities: List[str]) -> Dict[str, Any]:
        """
        Extract spatial relationships.
        
        Args:
            image_path: Path to image file
            entities: List of entity names to analyze
            
        Returns:
            Dictionary with spatial relationships
        """
        # Format prompt with entities
        entity_list = ", ".join(entities)
        prompt = f"{self.query_prompt}\n\nEntities: {entity_list}"
        
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"
            
        # Query model
        response = self.query(image_path, prompt, max_tokens=512)
        
        return {
            "entities": entities,
            "relationships": self._parse_relationships(response, entities),
            "raw_response": response
        }
        
    def _parse_relationships(self, response: str, entities: List[str]) -> List[Dict[str, str]]:
        """
        Parse relationships from response.
        
        This is simplified - production version would use more
        sophisticated NLP or structured output.
        """
        relationships = []
        
        # Simple pattern matching for common spatial relations
        spatial_words = ["left", "right", "above", "below", "front", "behind", "on", "in", "next to"]
        
        # Placeholder: In production, use proper NLP parsing
        # For now, return empty list (GBC will use detection-based fallback)
        
        return relationships


class QwenVLCompositionQuery(QwenVLBase):
    """
    Composition query for analyzing image composition.
    
    Analyzes overall composition, layout, and visual structure.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        system_file: Optional[str] = None,
        query_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        
        self.system_prompt = self._load_prompt(system_file) if system_file else ""
        self.query_prompt = self._load_prompt(query_file) if query_file else self._default_composition_prompt()
        
    def _load_prompt(self, file_path: str) -> str:
        """Load prompt from file."""
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logger.warning(f"Failed to load prompt from {file_path}: {e}")
            return ""
            
    def _default_composition_prompt(self) -> str:
        """Default prompt for composition analysis."""
        return """Analyze the composition of this image:
1. Overall layout and structure
2. Visual hierarchy and focal points
3. Balance and symmetry
4. Grouping of elements
5. Foreground, middle ground, background

Provide a structured analysis."""
        
    def __call__(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze image composition.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with composition analysis
        """
        prompt = self.query_prompt
        
        if self.system_prompt:
            prompt = f"{self.system_prompt}\n\n{prompt}"
            
        # Query model
        response = self.query(image_path, prompt, max_tokens=512)
        
        return {
            "composition_analysis": response,
            "raw_response": response
        }


# Convenience function for testing
def test_qwen_vl():
    """Test Qwen-VL queries."""
    print("=== Testing Qwen-VL Queries ===\n")
    
    try:
        # Test image query
        print("1. Testing Image Query...")
        image_query = QwenVLImageQuery()
        print("   ✅ QwenVLImageQuery initialized")
        
        # Test entity query
        print("2. Testing Entity Query...")
        entity_query = QwenVLEntityQuery()
        print("   ✅ QwenVLEntityQuery initialized")
        
        # Test relation query
        print("3. Testing Relation Query...")
        relation_query = QwenVLRelationQuery()
        print("   ✅ QwenVLRelationQuery initialized")
        
        # Test composition query
        print("4. Testing Composition Query...")
        composition_query = QwenVLCompositionQuery()
        print("   ✅ QwenVLCompositionQuery initialized")
        
        print("\n✅ All Qwen-VL query modules initialized successfully!")
        print("\nNote: Models will be loaded on first actual query.")
        print("      This requires ~14GB GPU memory for Qwen2-VL-7B.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to install dependencies:")
        print("  pip install qwen-vl-utils accelerate")


if __name__ == "__main__":
    test_qwen_vl()
