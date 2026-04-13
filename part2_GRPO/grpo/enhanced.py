import re
import numpy as np

def reasoning_preservation_reward_func(prompts, completions, answer, **kwargs):
    """
    Prevent GRPO from degrading SFT-learned reasoning capability.
    
    CRITICAL INSIGHT: Model already learned GSM8K-style reasoning in SFT.
    GRPO optimized it away because shortcuts were more reward-efficient.
    
    SOLUTION: Make full reasoning MORE REWARDING than shortcuts.
    
    Components:
    1. Length requirement: Prevent one-line shortcuts
    2. GSM8K pattern bonus: Reward <<calculation>> markers
    3. Multi-step bonus: Reward multiple reasoning steps
    4. Placeholder penalty: Harshly punish [placeholder] exploit
    5. Correctness reward: Increased weight to justify effort
    """
    rewards = []
    
    for completion, correct_answer in zip(completions, answer):
        # Extract completion text
        if isinstance(completion, list) and len(completion) > 0:
            text = completion[0].get('content', '') if isinstance(completion[0], dict) else str(completion[0])
        else:
            text = str(completion)
        
        total_reward = 0.0
        
        # ================================================================
        # Component 1: LENGTH-BASED QUALITY CHECK
        # GSM8K solutions are typically 50-150 tokens
        # Prevent degenerate short responses
        # ================================================================
        word_count = len(text.split())
        
        if word_count < 15:
            # Too short - likely a shortcut
            total_reward -= 1.5
        elif word_count < 25:
            # Short but might be valid for easy problems
            total_reward -= 0.5
        elif word_count > 200:
            # Too verbose - might be rambling
            total_reward -= 0.3
        # else: 25-200 words is good range
        
        # ================================================================
        # Component 2: GSM8K PATTERN PRESERVATION
        # Reward the EXACT patterns model learned in SFT
        # ================================================================
        gsm8k_bonus = 0.0
        
        # Pattern A: <<calculation>> markers (unique to GSM8K!)
        calc_markers = re.findall(r'<<[^>]+>>', text)
        if len(calc_markers) >= 1:
            gsm8k_bonus += 0.5
        if len(calc_markers) >= 2:
            gsm8k_bonus += 0.3  # Bonus for multiple steps
        
        # Pattern B: Explicit calculations shown
        # Format: "X op Y = Z"
        explicit_calcs = re.findall(r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+', text)
        if len(explicit_calcs) >= 1:
            gsm8k_bonus += 0.4
        if len(explicit_calcs) >= 2:
            gsm8k_bonus += 0.3
        
        # Pattern C: Multi-line structure (GSM8K solutions span lines)
        line_count = len([l for l in text.split('\n') if l.strip()])
        if line_count >= 2:
            gsm8k_bonus += 0.3
        
        total_reward += gsm8k_bonus
        
        # ================================================================
        # Component 3: CALCULATION CORRECTNESS VERIFICATION
        # Actually check if shown calculations are correct
        # This is HARD to game - must do real math
        # ================================================================
        calc_pattern = r'(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)'
        calculations = re.findall(calc_pattern, text)
        
        correct_calc_count = 0
        for a, op, b, result in calculations:
            try:
                a, b, result = float(a), float(b), float(result)
                expected = {
                    '+': a + b,
                    '-': a - b,
                    '*': a * b,
                    '/': a / b if b != 0 else float('inf')
                }.get(op)
                
                if expected is not None and abs(expected - result) < 0.01:
                    correct_calc_count += 1
            except:
                pass
        
        if correct_calc_count >= 1:
            total_reward += 0.6
        if correct_calc_count >= 2:
            total_reward += 0.4  # Extra for multiple correct steps
        
        # ================================================================
        # Component 4: PLACEHOLDER DETECTION & HARSH PENALTY
        # This is the EXACT exploit we discovered
        # Must be punished severely to prevent
        # ================================================================
        placeholder_patterns = [
            r'\[(?:total|sum|answer|result|profit|cost|price|amount|difference|product|quotient|value)\]',
            r'the answer is \[[^\]]+\]',
            r'\[.*?(?:number|count|quantity).*?\]',
        ]
        
        has_placeholder = any(re.search(p, text.lower()) for p in placeholder_patterns)
        
        if has_placeholder:
            total_reward -= 3.0  # SEVERE PENALTY - make shortcuts unprofitable
        
        # ================================================================
        # Component 5: FORMAT COMPLIANCE (Reduced Weight)
        # Still want "the answer is", but it's less important now
        # ================================================================
        if "the answer is" in text.lower() or "####" in text:
            total_reward += 0.2  # Reduced from 0.5
        
        # ================================================================
        # Component 6: CORRECTNESS (Increased Weight)
        # Make getting the right answer VERY valuable
        # Must justify the effort of showing work
        # ================================================================
        
        # Try to extract final answer
        # Support both "The answer is X" and "#### X" formats
        answer_match = re.search(
            r'(?:the answer is|####)\s*:?\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)',
            text.lower()
        )
        
        if answer_match:
            predicted = answer_match.group(1).replace(',', '')
            correct_str = str(correct_answer).replace(',', '')
            
            try:
                if abs(float(predicted) - float(correct_str)) < 0.01:
                    total_reward += 4.0  # INCREASED from 2.0 - make correctness very valuable
                else:
                    total_reward -= 1.0  # Wrong answer penalty
            except ValueError:
                total_reward -= 1.0
        else:
            # No answer extracted at all
            total_reward -= 1.0
        
        rewards.append(total_reward)
    
    return rewards


def analyze_response_quality(text):
    """
    Helper function for logging/debugging.
    Analyzes what patterns are present in a response.
    """
    return {
        'word_count': len(text.split()),
        'line_count': len([l for l in text.split('\n') if l.strip()]),
        'has_calc_markers': bool(re.search(r'<<[^>]+>>', text)),
        'calc_marker_count': len(re.findall(r'<<[^>]+>>', text)),
        'explicit_calc_count': len(re.findall(r'\d+\s*[+\-*/]\s*\d+\s*=\s*\d+', text)),
        'has_placeholder': bool(re.search(r'\[(?:total|sum|answer)', text.lower())),
        'has_answer_format': bool(re.search(r'the answer is|####', text.lower())),
    }