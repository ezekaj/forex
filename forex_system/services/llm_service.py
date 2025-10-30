"""
LLM service for signal review and risk management.

Uses OpenAI or Anthropic to review ML-generated trading signals
by analyzing news context and providing reasoning.

Inspired by AI-Trader's agent-based decision making.
"""
import os
import json
from datetime import datetime
from typing import Dict, Optional, Literal
from dataclasses import dataclass
import anthropic
import openai


@dataclass
class SignalReview:
    """LLM's review of a trading signal."""
    decision: Literal['APPROVE', 'REJECT', 'MODIFY']
    confidence: float  # 0.0 to 1.0
    reasoning: str
    suggested_position_size: Optional[float] = None
    stop_loss_adjustment: Optional[float] = None
    take_profit_adjustment: Optional[float] = None
    cost_usd: float = 0.0  # Track LLM API cost


class LLMService:
    """Service for LLM-based signal review."""

    # Cost per 1K tokens (as of 2025)
    COSTS = {
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-4o': {'input': 0.005, 'output': 0.015},
        'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
        'claude-3-7-sonnet': {'input': 0.003, 'output': 0.015},
        'claude-3-5-haiku': {'input': 0.001, 'output': 0.005},
    }

    SYSTEM_PROMPT = """You are an expert forex trading risk manager and analyst.

Your role: Review ML-generated trading signals by analyzing:
1. News sentiment and fundamental context
2. Market conditions and volatility
3. Risk/reward assessment
4. Potential catalysts or risks

You must decide: APPROVE, REJECT, or MODIFY the signal.

Guidelines:
- APPROVE: Signal aligns with fundamentals, good risk/reward, clear catalyst
- REJECT: Signal contradicts news, high event risk, poor timing
- MODIFY: Signal has merit but needs risk adjustment (size, stops, targets)

Be conservative. It's better to reject a mediocre trade than approve a risky one.
Your goal: Improve win rate by filtering bad signals and catching Black Swans.

Always provide clear, concise reasoning (2-3 sentences max)."""

    def __init__(
        self,
        provider: Literal['openai', 'anthropic'] = 'anthropic',
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize LLM service.

        Args:
            provider: LLM provider (openai or anthropic)
            model: Model name (defaults to best value model)
            api_key: API key (reads from env if None)
        """
        self.provider = provider
        self.total_cost = 0.0  # Track cumulative cost

        if provider == 'openai':
            self.model = model or 'gpt-4o-mini'  # Best value
            self.api_key = api_key or os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not set")
            openai.api_key = self.api_key

        elif provider == 'anthropic':
            self.model = model or 'claude-3-5-haiku-20241022'  # Best value
            self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self.client = anthropic.Anthropic(api_key=self.api_key)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def review_signal(
        self,
        signal: Dict,
        market_context: Dict,
        price: float,
        pair: str
    ) -> SignalReview:
        """
        Review a trading signal using LLM.

        Args:
            signal: ML model's signal (direction, confidence, etc.)
            market_context: News and fundamental context from NewsService
            price: Current market price
            pair: Currency pair

        Returns:
            SignalReview with LLM's decision
        """
        # Build prompt
        prompt = self._build_review_prompt(signal, market_context, price, pair)

        # Call LLM
        if self.provider == 'openai':
            response = self._call_openai(prompt)
        else:
            response = self._call_anthropic(prompt)

        # Parse response
        review = self._parse_response(response)

        # Track cost
        self.total_cost += review.cost_usd

        return review

    def _build_review_prompt(
        self,
        signal: Dict,
        market_context: Dict,
        price: float,
        pair: str
    ) -> str:
        """
        Build review prompt for LLM.

        Args:
            signal: ML signal
            market_context: News context
            price: Current price
            pair: Currency pair

        Returns:
            Prompt string
        """
        # Extract signal details
        direction = signal.get('direction', 'UNKNOWN')  # BUY, SELL, HOLD
        ml_confidence = signal.get('confidence', 0.0)
        features_summary = signal.get('features_summary', {})

        # Extract market context
        base_sentiment = market_context['base_currency']['sentiment']
        quote_sentiment = market_context['quote_currency']['sentiment']
        news_volume = market_context['news_volume_24h']
        headlines = market_context['recent_headlines']

        # Format headlines
        headlines_text = "\n".join([
            f"- {h['title']} ({h['source']})"
            for h in headlines[:5]
        ])

        prompt = f"""Review this forex trading signal:

**SIGNAL DETAILS**
Pair: {pair}
Direction: {direction}
ML Confidence: {ml_confidence:.1%}
Current Price: {price:.5f}

**TECHNICAL ANALYSIS (ML Features)**
{self._format_features(features_summary)}

**FUNDAMENTAL CONTEXT**
Base Currency ({market_context['base_currency']['code']}):
- Sentiment: {base_sentiment:.2f} (-1=bearish, +1=bullish)
- News Articles (7 days): {market_context['base_currency']['article_count']}
- Key Topics: {', '.join(market_context['base_currency']['topics'][:3])}

Quote Currency ({market_context['quote_currency']['code']}):
- Sentiment: {quote_sentiment:.2f}
- News Articles (7 days): {market_context['quote_currency']['article_count']}
- Key Topics: {', '.join(market_context['quote_currency']['topics'][:3])}

News Volume (24h): {news_volume} articles

**RECENT HEADLINES**
{headlines_text if headlines_text else 'No recent headlines found'}

**YOUR TASK**
Decide: APPROVE, REJECT, or MODIFY this signal.

Respond in JSON format:
{{
  "decision": "APPROVE|REJECT|MODIFY",
  "confidence": 0.85,
  "reasoning": "Brief explanation (2-3 sentences)",
  "suggested_position_size": 1.0,
  "stop_loss_adjustment": 0.0,
  "take_profit_adjustment": 0.0
}}

Notes:
- position_size: 1.0 = full position, 0.5 = half position, 0 = reject
- adjustments: in pips (0 = no change, positive = widen, negative = tighten)
"""
        return prompt

    def _format_features(self, features_summary: Dict) -> str:
        """Format feature summary for prompt."""
        if not features_summary:
            return "No feature summary available"

        lines = []
        for key, value in features_summary.items():
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.3f}")
            else:
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) if lines else "No features available"

    def _call_openai(self, prompt: str) -> Dict:
        """
        Call OpenAI API.

        Args:
            prompt: Review prompt

        Returns:
            Dict with response and usage
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistent decisions
                max_tokens=500
            )

            content = response.choices[0].message.content
            usage = response.usage

            # Calculate cost
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)

            return {
                'content': content,
                'cost_usd': cost,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }

        except Exception as e:
            print(f"Error calling OpenAI: {str(e)}")
            # Return safe default (reject)
            return {
                'content': '{"decision": "REJECT", "confidence": 0.0, "reasoning": "LLM API error"}',
                'cost_usd': 0.0,
                'input_tokens': 0,
                'output_tokens': 0
            }

    def _call_anthropic(self, prompt: str) -> Dict:
        """
        Call Anthropic API.

        Args:
            prompt: Review prompt

        Returns:
            Dict with response and usage
        """
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.3,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.content[0].text
            usage = response.usage

            # Calculate cost
            input_tokens = usage.input_tokens
            output_tokens = usage.output_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)

            return {
                'content': content,
                'cost_usd': cost,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens
            }

        except Exception as e:
            print(f"Error calling Anthropic: {str(e)}")
            # Return safe default (reject)
            return {
                'content': '{"decision": "REJECT", "confidence": 0.0, "reasoning": "LLM API error"}',
                'cost_usd': 0.0,
                'input_tokens': 0,
                'output_tokens': 0
            }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate API call cost.

        Args:
            input_tokens: Input token count
            output_tokens: Output token count

        Returns:
            Cost in USD
        """
        # Get model costs (normalize model names)
        model_key = self.model
        if 'gpt-4o-mini' in model_key:
            model_key = 'gpt-4o-mini'
        elif 'gpt-4o' in model_key:
            model_key = 'gpt-4o'
        elif 'gpt-4' in model_key:
            model_key = 'gpt-4-turbo'
        elif 'haiku' in model_key:
            model_key = 'claude-3-5-haiku'
        elif 'sonnet' in model_key:
            model_key = 'claude-3-7-sonnet'

        costs = self.COSTS.get(model_key, {'input': 0.01, 'output': 0.03})

        input_cost = (input_tokens / 1000) * costs['input']
        output_cost = (output_tokens / 1000) * costs['output']

        return input_cost + output_cost

    def _parse_response(self, response: Dict) -> SignalReview:
        """
        Parse LLM response into SignalReview.

        Args:
            response: LLM API response

        Returns:
            SignalReview object
        """
        content = response['content']
        cost = response['cost_usd']

        try:
            # Try to parse JSON
            # LLMs sometimes wrap JSON in markdown code blocks
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]

            data = json.loads(content)

            return SignalReview(
                decision=data.get('decision', 'REJECT'),
                confidence=float(data.get('confidence', 0.0)),
                reasoning=data.get('reasoning', 'No reasoning provided'),
                suggested_position_size=data.get('suggested_position_size'),
                stop_loss_adjustment=data.get('stop_loss_adjustment'),
                take_profit_adjustment=data.get('take_profit_adjustment'),
                cost_usd=cost
            )

        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            print(f"Response content: {content}")

            # Return safe default
            return SignalReview(
                decision='REJECT',
                confidence=0.0,
                reasoning=f'Failed to parse LLM response: {str(e)}',
                cost_usd=cost
            )

    def get_total_cost(self) -> float:
        """Get total cost of all LLM calls."""
        return self.total_cost

    def reset_cost(self) -> None:
        """Reset cost tracker."""
        self.total_cost = 0.0
