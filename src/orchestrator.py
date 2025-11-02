"""
AI Orchestrator - Main entry point for GenAI platform
Manages multiple AI models and routes requests
"""
import logging
from typing import Dict, List, Optional
import time

from .models.openai_client import OpenAIClient
from .models.claude_client import ClaudeClient
from .models.gemini_client import GeminiClient
from .rag.retriever import RAGRetriever
from .agents.researcher_agent import ResearcherAgent
from .prompts.prompt_manager import PromptManager
from .utils.monitoring import MetricsCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIOrchestrator:
    """
    Main orchestrator for GenAI platform
    Routes requests to appropriate models and services
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize AI Orchestrator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize model clients
        self.openai_client = OpenAIClient(self.config)
        self.claude_client = ClaudeClient(self.config)
        self.gemini_client = GeminiClient(self.config)
        
        # Initialize components
        self.rag_retriever = RAGRetriever(self.config)
        self.prompt_manager = PromptManager()
        self.metrics = MetricsCollector()
        
        # Model routing configuration
        self.model_capabilities = {
            'gpt-4': {'cost': 'high', 'quality': 'excellent', 'speed': 'medium'},
            'gpt-3.5-turbo': {'cost': 'low', 'quality': 'good', 'speed': 'fast'},
            'claude-3': {'cost': 'high', 'quality': 'excellent', 'speed': 'medium'},
            'gemini-pro': {'cost': 'medium', 'quality': 'good', 'speed': 'fast'}
        }
        
        logger.info("AI Orchestrator initialized successfully")
    
    def generate(
        self,
        prompt: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict:
        """
        Generate text using specified model
        
        Args:
            prompt: Input prompt
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generation result with metadata
        """
        start_time = time.time()
        
        try:
            # Route to appropriate model
            if model.startswith('gpt'):
                response = self.openai_client.generate(
                    prompt, model, temperature, max_tokens
                )
            elif model.startswith('claude'):
                response = self.claude_client.generate(
                    prompt, model, temperature, max_tokens
                )
            elif model.startswith('gemini'):
                response = self.gemini_client.generate(
                    prompt, model, temperature, max_tokens
                )
            else:
                raise ValueError(f"Unsupported model: {model}")
            
            # Track metrics
            self.metrics.record_generation(
                model=model,
                tokens=response.get('tokens_used', 0),
                latency=time.time() - start_time,
                cost=response.get('cost', 0)
            )
            
            return {
                'success': True,
                'text': response['text'],
                'model': model,
                'tokens_used': response.get('tokens_used', 0),
                'latency': time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model
            }
    
    def generate_with_rag(
        self,
        query: str,
        knowledge_base: str,
        model: str = "gpt-4",
        top_k: int = 5
    ) -> Dict:
        """
        Generate response with RAG (Retrieval Augmented Generation)
        
        Args:
            query: User query
            knowledge_base: Name of knowledge base
            model: Model to use for generation
            top_k: Number of documents to retrieve
            
        Returns:
            Generated response with sources
        """
        logger.info(f"RAG query: {query}")
        
        try:
            # Retrieve relevant documents
            retrieved_docs = self.rag_retriever.retrieve(
                query=query,
                knowledge_base=knowledge_base,
                top_k=top_k
            )
            
            # Build context from retrieved documents
            context = self._build_context(retrieved_docs)
            
            # Generate prompt with context
            rag_prompt = self.prompt_manager.get_rag_prompt(
                query=query,
                context=context
            )
            
            # Generate response
            response = self.generate(
                prompt=rag_prompt,
                model=model,
                temperature=0.3  # Lower temperature for factual responses
            )
            
            if response['success']:
                response['sources'] = [
                    {
                        'text': doc['text'][:200],
                        'score': doc['score'],
                        'metadata': doc.get('metadata', {})
                    }
                    for doc in retrieved_docs
                ]
            
            return response
            
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def execute_agent_task(
        self,
        task: str,
        agents: List[str],
        **kwargs
    ) -> Dict:
        """
        Execute multi-agent task
        
        Args:
            task: Task description
            agents: List of agent names to use
            
        Returns:
            Task execution result
        """
        logger.info(f"Executing agent task: {task}")
        
        results = []
        
        for agent_name in agents:
            agent = self._get_agent(agent_name)
            if agent:
                result = agent.execute(task, **kwargs)
                results.append({
                    'agent': agent_name,
                    'result': result
                })
        
        return {
            'success': True,
            'task': task,
            'results': results
        }
    
    def optimize_prompt(
        self,
        prompt: str,
        objective: str = "clarity"
    ) -> str:
        """
        Optimize prompt using AI
        
        Args:
            prompt: Original prompt
            objective: Optimization objective
            
        Returns:
            Optimized prompt
        """
        optimization_prompt = f"""Optimize the following prompt for {objective}:

Original Prompt:
{prompt}

Provide an improved version that is clearer, more specific, and likely to produce better results."""
        
        response = self.generate(
            prompt=optimization_prompt,
            model="gpt-4",
            temperature=0.5
        )
        
        return response.get('text', prompt)
    
    def compare_models(
        self,
        prompt: str,
        models: List[str]
    ) -> Dict:
        """
        Compare responses from multiple models
        
        Args:
            prompt: Test prompt
            models: List of models to compare
            
        Returns:
            Comparison results
        """
        results = {}
        
        for model in models:
            response = self.generate(prompt=prompt, model=model)
            results[model] = response
        
        return {
            'prompt': prompt,
            'models_compared': models,
            'results': results
        }
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context string from retrieved documents"""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[{i}] {doc['text']}")
        return "\n\n".join(context_parts)
    
    def _get_agent(self, agent_name: str):
        """Get agent instance by name"""
        if agent_name == "researcher":
            return ResearcherAgent(self)
        # Add more agents as needed
        return None
    
    def get_metrics(self) -> Dict:
        """Get platform metrics"""
        return self.metrics.get_summary()
