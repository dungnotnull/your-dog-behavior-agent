"""
LLM-powered interpretation engine for dog behavior classification results.
Generates natural-language explanations, breed context, and owner guidance.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from tools.faiss_index_manager import KnowledgeIndexManager

logger = logging.getLogger(__name__)

SECOND_KNOWLEDGE_BRAIN_PATH = Path(__file__).parent.parent.parent / "SECOND-KNOWLEDGE-BRAIN.md"

INTERPRETATION_PROMPT = """You are an expert in canine ethology and applied animal behavior science.
A dog vocalization recording has been analyzed by an ML classifier.

Classification: {behavior_label} (confidence: {confidence:.0%})
Secondary signal: {secondary_label} ({secondary_confidence:.0%})
Acoustic features: centroid={spectral_centroid:.0f}Hz, rms_energy={rms_energy:.4f}, tempo={tempo:.1f}bpm, zcr={zcr:.4f}
Dog profile: breed={breed}, age={age}
Research context: {research_context}

Respond ONLY in valid JSON with exactly these keys:
{{
  "narrative": "2-3 sentence plain-language explanation of what the dog is communicating",
  "breed_context": "one sentence breed-specific context if relevant (empty string if breed unknown or irrelevant)",
  "guidance_steps": ["step 1 for the owner", "step 2", "step 3"],
  "urgency_level": "low",
  "confidence_explanation": "one sentence about what acoustic features drove this classification"
}}
urgency_level must be exactly one of: low, medium, high"""

FALLBACK_INTERPRETATIONS = {
    "aggression": {
        "narrative": "Your dog is showing signs of aggression or territorial behavior. Low-pitched, harsh vocalizations indicate discomfort, resource guarding, or threat perception. This requires careful management.",
        "breed_context": "",
        "guidance_steps": [
            "Remove your dog from the triggering situation immediately and calmly",
            "Do not punish the vocalization — identify and address the underlying trigger",
            "Consult a certified professional dog trainer or applied animal behaviorist",
        ],
        "urgency_level": "high",
        "confidence_explanation": "Low spectral centroid and high energy are acoustic signatures of aggressive vocalizations.",
    },
    "fear": {
        "narrative": "Your dog appears fearful or anxious. These high-pitched, softer vocalizations signal distress and a desire to escape or seek safety. The trigger should be identified and managed.",
        "breed_context": "",
        "guidance_steps": [
            "Allow your dog to move away from the source of fear — never force approach",
            "Create a calm, safe space where your dog can decompress",
            "Consider desensitization and counter-conditioning with a professional trainer",
        ],
        "urgency_level": "medium",
        "confidence_explanation": "High-pitched, low-energy vocalizations with narrow bandwidth match fear/appeasement patterns.",
    },
    "excitement": {
        "narrative": "Your dog is expressing positive excitement or anticipation. Rapid, high-pitched barks with high energy are typical of anticipatory arousal, such as before a walk, meal, or play.",
        "breed_context": "",
        "guidance_steps": [
            "Wait for a calmer moment before rewarding or providing the anticipated item",
            "Teach a 'settle' or 'calm' cue to help manage excitement levels",
            "Provide adequate exercise and mental stimulation to reduce baseline arousal",
        ],
        "urgency_level": "low",
        "confidence_explanation": "High tempo, high spectral centroid, and high RMS energy indicate excitement arousal.",
    },
    "pain_distress": {
        "narrative": "Your dog may be experiencing pain or significant distress. Sustained, high-pitched vocalizations of this type require immediate attention to rule out injury or illness.",
        "breed_context": "",
        "guidance_steps": [
            "Check your dog for visible signs of injury, limping, or physical distress",
            "Contact your veterinarian immediately if the vocalizations persist",
            "Do not attempt to handle a dog in pain without proper precautions to avoid biting",
        ],
        "urgency_level": "high",
        "confidence_explanation": "Sustained high-pitch, consistent frequency, and mid-range energy match pain/distress vocalization patterns.",
    },
    "attention_seeking": {
        "narrative": "Your dog is seeking attention or interaction. These regular, medium-pitched vocalizations are a learned behavior — your dog has discovered this is an effective way to get your response.",
        "breed_context": "",
        "guidance_steps": [
            "Avoid rewarding attention-seeking barking with immediate attention — wait for quiet",
            "Ensure your dog is getting adequate exercise, play, and mental enrichment daily",
            "Teach an incompatible quiet behavior and reward calm interaction instead",
        ],
        "urgency_level": "low",
        "confidence_explanation": "Regular interval, medium pitch, and moderate energy match attention-seeking bark patterns.",
    },
    "play": {
        "narrative": "Your dog is inviting play or expressing playful excitement. Short, higher-pitched staccato vocalizations are a positive social signal indicating a relaxed and affiliative state.",
        "breed_context": "",
        "guidance_steps": [
            "Engage with appropriate play — fetch, tug, or interactive toys",
            "Monitor play arousal levels to prevent escalation",
            "Ensure both parties (dogs or human-dog) are enjoying the interaction equally",
        ],
        "urgency_level": "low",
        "confidence_explanation": "Short staccato pattern, higher pitch, and moderate energy are acoustic signatures of play vocalizations.",
    },
    "alert_warning": {
        "narrative": "Your dog has detected something novel in the environment and is alerting you. This sentinel behavior is normal and typically directed at unfamiliar sounds, people, or animals.",
        "breed_context": "",
        "guidance_steps": [
            "Acknowledge your dog calmly ('thank you') then redirect their attention",
            "Investigate the alerting stimulus if appropriate to address the underlying trigger",
            "Teach a 'quiet' cue and reward cessation of barking after initial alert",
        ],
        "urgency_level": "low",
        "confidence_explanation": "Single or few deep barks with low spectral centroid and high initial energy match alert/warning patterns.",
    },
    "greeting": {
        "narrative": "Your dog is expressing a social greeting. These mixed-pitch, moderate-energy vocalizations are affiliative signals associated with the approach of familiar individuals.",
        "breed_context": "",
        "guidance_steps": [
            "Greet your dog calmly to avoid reinforcing over-excited greeting behavior",
            "Wait for all four paws on the floor before providing affection",
            "This is normal, positive behavior — enjoy the connection with your dog",
        ],
        "urgency_level": "low",
        "confidence_explanation": "Mixed pitch, moderate energy, and varied tempo are characteristic of social greeting vocalizations.",
    },
    "uncertain": {
        "narrative": "The acoustic analysis could not confidently classify this vocalization. Multiple behavioral signals may be present, or the audio quality may be insufficient for accurate classification.",
        "breed_context": "",
        "guidance_steps": [
            "Try recording a longer, clearer audio clip closer to the dog",
            "Observe accompanying body language: tail position, ear orientation, and body posture",
            "Consult a certified animal behavior professional for in-person assessment",
        ],
        "urgency_level": "low",
        "confidence_explanation": "Low classifier confidence across all classes indicates ambiguous acoustic features.",
    },
}


@dataclass
class InterpretationResult:
    narrative: str
    breed_context: str
    guidance_steps: List[str]
    urgency_level: str       # "low" | "medium" | "high"
    confidence_explanation: str
    research_context_used: bool
    llm_provider_used: str   # "claude" | "openai" | "ollama" | "fallback"


class InterpretationEngine:
    """Generates LLM-powered natural-language interpretation of dog behavior classifications."""

    def __init__(self):
        self._llm = None
        self._hf = None
        self._knowledge_index = None  # FAISS index once built

    def _get_llm(self):
        if self._llm is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from tools.llm_client import UnifiedLLMClient
            self._llm = UnifiedLLMClient()
        return self._llm

    def _get_hf(self):
        if self._hf is None:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from tools.hf_model_manager import HFModelManager
            self._hf = HFModelManager.instance()
        return self._hf

    def _get_knowledge_index(self):
        if self._knowledge_index is None:
            self._knowledge_index = KnowledgeIndexManager()
        return self._knowledge_index

    def _load_papers_from_brain(self) -> List[Dict[str, str]]:
        """Parse the Key Research Papers table from SECOND-KNOWLEDGE-BRAIN.md."""
        papers = []
        if not SECOND_KNOWLEDGE_BRAIN_PATH.exists():
            return papers
        content = SECOND_KNOWLEDGE_BRAIN_PATH.read_text(encoding="utf-8")
        in_table = False
        header_passed = False
        for line in content.splitlines():
            if "| # | Title |" in line:
                in_table = True
                continue
            if in_table and line.startswith("|---"):
                header_passed = True
                continue
            if in_table and header_passed and line.startswith("|"):
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) >= 7:
                    papers.append({
                        "title": parts[1],
                        "authors": parts[2],
                        "year": parts[3],
                        "key_finding": parts[6],
                        "relevance": parts[7] if len(parts) > 7 else "",
                    })
            elif in_table and header_passed and not line.startswith("|"):
                break
        return papers

    def _retrieve_relevant_context(self, behavior_label: str, acoustic_summary: str) -> str:
        """
        Retrieve relevant research context using FAISS + BGE first,
        then fall back to in-memory brute-force BGE cosine search.
        """
        query = f"{behavior_label} dog behavior {acoustic_summary}"

        # Primary: FAISS index search
        try:
            idx = self._get_knowledge_index()
            papers = idx.search(query, top_k=3)
            if papers:
                combined = " ".join(
                    f"{p['title']} ({p['year']}): {p['key_finding']}" for p in papers
                )
                return self._get_hf().summarize(combined[:1024])
        except Exception as exc:
            logger.debug("FAISS knowledge search failed, falling back to brute-force BGE: %s", exc)

        # Fallback: brute-force BGE cosine search over SECOND-KNOWLEDGE-BRAIN.md
        papers = self._load_papers_from_brain()
        if not papers:
            return ""

        try:
            hf = self._get_hf()
            query_emb = hf.encode(query)
            paper_texts = [f"{p['title']} {p['key_finding']}" for p in papers]
            paper_embs = hf.encode_batch(paper_texts)

            import numpy as np
            query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
            paper_norms = paper_embs / (np.linalg.norm(paper_embs, axis=1, keepdims=True) + 1e-8)
            scores = paper_norms @ query_norm
            top_indices = np.argsort(scores)[::-1][:3]
            top_papers = [papers[i] for i in top_indices]

            combined = " ".join(
                f"{p['title']} ({p['year']}): {p['key_finding']}" for p in top_papers
            )
            return hf.summarize(combined[:1024])
        except Exception:
            # Last-resort fallback: return first paper's key finding
            return papers[0]["key_finding"][:200]

    def interpret(
        self,
        classification_result,
        audio_features,
        breed: str = "unknown",
        age: str = "unknown",
    ) -> InterpretationResult:
        """
        Generate natural-language interpretation via LLM.
        Falls back to template if all LLM providers fail.
        """
        label = classification_result.behavior_label
        acoustic_summary = f"centroid={audio_features.spectral_centroid:.0f}Hz"
        research_ctx = self._retrieve_relevant_context(label, acoustic_summary)

        prompt = INTERPRETATION_PROMPT.format(
            behavior_label=label,
            confidence=classification_result.confidence,
            secondary_label=classification_result.secondary_label,
            secondary_confidence=classification_result.secondary_confidence,
            spectral_centroid=audio_features.spectral_centroid,
            rms_energy=audio_features.rms_energy,
            tempo=audio_features.tempo,
            zcr=audio_features.zero_crossing_rate,
            breed=breed,
            age=age,
            research_context=research_ctx[:500] if research_ctx else "No research context available.",
        )

        provider_used = "fallback"
        parsed = None

        try:
            llm = self._get_llm()
            response = llm.complete(prompt, max_tokens=600)
            parsed = self._parse_json_response(response)
            provider_used = llm.last_provider_used
        except Exception as exc:
            logger.warning("LLM interpretation failed, using fallback template: %s", exc)

        if parsed is None:
            parsed = FALLBACK_INTERPRETATIONS.get(
                label, FALLBACK_INTERPRETATIONS["uncertain"]
            ).copy()
            provider_used = "fallback"

        # Safety gates
        urgency = parsed.get("urgency_level", "low")
        guidance = parsed.get("guidance_steps", ["Observe your dog's behavior carefully."])
        if label == "pain_distress" and classification_result.confidence > 0.60:
            urgency = "high"
            if not any("vet" in s.lower() or "veterinar" in s.lower() for s in guidance):
                guidance.insert(0, "URGENT: Contact your veterinarian immediately for professional evaluation.")
        elif label == "aggression" and classification_result.confidence > 0.65:
            urgency = "high"
            if not any("safe" in s.lower() or "professional" in s.lower() for s in guidance):
                guidance.insert(0, "SAFETY: Keep people and other animals away from the dog until calm.")

        return InterpretationResult(
            narrative=parsed.get("narrative", ""),
            breed_context=parsed.get("breed_context", ""),
            guidance_steps=guidance[:3],
            urgency_level=urgency,
            confidence_explanation=parsed.get("confidence_explanation", ""),
            research_context_used=bool(research_ctx),
            llm_provider_used=provider_used,
        )

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response text."""
        # Strip markdown fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find a JSON object in the text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    return None
            return None

    def format_result(
        self, result: InterpretationResult, classification_result, include_probabilities: bool = False
    ) -> str:
        """Format the full analysis as human-readable text."""
        urgency_icon = {"low": "✓", "medium": "⚠", "high": "⚠⚠"}.get(result.urgency_level, "")
        lines = [
            f"┌─ Dog Behavior Analysis {'─' * 40}",
            f"│ Behavior: {classification_result.behavior_label.upper().replace('_', ' ')} "
            f"(confidence: {classification_result.confidence:.0%}) {urgency_icon}",
            f"│ Secondary: {classification_result.secondary_label} ({classification_result.secondary_confidence:.0%})",
            f"│ Model: {classification_result.model_used}",
            "├─ Interpretation " + "─" * 44,
            f"│ {result.narrative}",
        ]
        if result.breed_context:
            lines.append(f"│ Breed context: {result.breed_context}")
        lines += [
            "├─ Owner Guidance " + "─" * 44,
        ]
        for i, step in enumerate(result.guidance_steps, 1):
            lines.append(f"│ {i}. {step}")
        if result.confidence_explanation:
            lines += [
                "├─ Why this classification " + "─" * 36,
                f"│ {result.confidence_explanation}",
            ]
        if include_probabilities:
            lines += ["├─ All Probabilities " + "─" * 41]
            for label, prob in sorted(
                classification_result.all_probabilities.items(), key=lambda x: -x[1]
            ):
                bar = "█" * int(prob * 20)
                lines.append(f"│ {label:<20} {bar:<20} {prob:.1%}")
        lines.append("└" + "─" * 61)
        return "\n".join(lines)
