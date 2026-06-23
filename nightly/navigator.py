

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

@dataclass
class NavigatorOutput:
    

    intent:        str       = "maintain current position"
    focus_objects: list[str] = field(default_factory=list)
    confidence:    float     = 1.0

    

    @classmethod
    def from_dict(cls, d: dict) -> "NavigatorOutput":
        
        objects = d.get("focus_objects", [])
        if not isinstance(objects, list):
            objects = []
        objects = [str(o) for o in objects[:3]]

        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 1.0))))
        except (TypeError, ValueError):
            conf = 1.0

        return cls(
            intent        = str(d.get("intent", "maintain current position"))[:96],
            focus_objects = objects,
            confidence    = conf,
        )

    @classmethod
    def identity(cls) -> "NavigatorOutput":
        
        return cls()

    def to_enriched_string(self) -> str:
        

        if self.focus_objects:
            objects_str = ", ".join(self.focus_objects)
            return f"{self.intent}; objects: {objects_str}"
        return self.intent

    def to_vector(self) -> torch.Tensor:
        

        return torch.tensor([self.confidence], dtype=torch.float32)

    VECTOR_DIM: int = 1

class StructuredProjector(nn.Module):
    

    def __init__(
        self,
        base_text_dim: int = 384,
        out_dim:       int = 384,
        hidden:        int = 256,
    ):
        super().__init__()
        inp = base_text_dim + NavigatorOutput.VECTOR_DIM   
        self.net = nn.Sequential(
            nn.Linear(inp,    hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        
        with torch.no_grad():
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        text_emb: torch.Tensor,   
        num_vec:  torch.Tensor,   
    ) -> torch.Tensor:
        x = torch.cat([text_emb, num_vec], dim=-1)   
        return self.net(x).unsqueeze(1)               

_SYSTEM_PROMPT = """\
You are the high-level navigator of an autonomous FPV drone.
A separate low-level SafetyNet running at 50 Hz handles all collision avoidance.
Your only job is to translate the pilot's command and the current flight context
into a clear semantic intent.

Think step by step about what the drone should be doing, then output a single
JSON object on the last line — no markdown, no explanation outside the JSON:

{
  "intent":        <string, imperative sentence ≤ 16 words describing the goal>,
  "focus_objects": <list of up to 3 strings: most relevant obstacles or targets>,
  "confidence":    <float in [0, 1]: how clearly you understand the situation>
}

Rules:
- Do NOT include roll, pitch, yaw, throttle, urgency or safety fields.
  The low-level pilot and SafetyNet handle all of that.
- "intent" should describe WHAT to achieve, not HOW to fly.
- Low confidence (< 0.4) means the situation is ambiguous or the command
  is unclear — the system will reduce language influence automatically.
- Output ONLY the JSON object on the last line of your response.
"""

class ReasoningLLM:
    

    _FALLBACK_MODELS = [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
    ]

    def __init__(
        self,
        model_id:       str = "Qwen/Qwen2.5-3B-Instruct",
        device:         str = "mps",
        max_new_tokens: int = 400,
        dtype:          str = "float16",
    ):
        self._model_id       = model_id
        self._device         = device
        self._max_new_tokens = max_new_tokens
        self._dtype          = getattr(torch, dtype)
        self._model          = None
        self._tokenizer      = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype       = self._dtype,
            trust_remote_code = True,
            device_map        = self._device,
        ).eval()

    def reason(self, user_message: str) -> str:
        
        self._ensure_loaded()
        messages = [
            {"role": "system",  "content": _SYSTEM_PROMPT},
            {"role": "user",    "content": user_message},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens = self._max_new_tokens,
                do_sample      = False,
                temperature    = 1.0,
            )
        out_ids = ids[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(out_ids, skip_special_tokens=True)

def _extract_json(raw: str) -> dict:
    
    matches = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    if not matches:
        return {}
    try:
        return json.loads(matches[-1])
    except json.JSONDecodeError:
        return {}

def build_telemetry_prompt(
    command:      str,
    speed_mps:    float,
    roll_deg:     float,
    pitch_deg:    float,
    yaw_deg:      float,
    altitude_m:   float,
    last_intent:  str = "",
) -> str:
    

    ctx = (
        f"Command: \"{command}\"\n\n"
        f"Telemetry:\n"
        f"  speed     = {speed_mps:.1f} m/s\n"
        f"  roll      = {roll_deg:+.1f}°\n"
        f"  pitch     = {pitch_deg:+.1f}°\n"
        f"  yaw       = {yaw_deg:.1f}°\n"
        f"  altitude  = {altitude_m:.1f} m\n"
    )
    if last_intent:
        ctx += f"\nPrevious intent: \"{last_intent}\"\n"
    ctx += "\nThink step by step, then output the JSON."
    return ctx

class ReasoningNavigator:
    

    def __init__(
        self,
        llm:       ReasoningLLM,
        text_enc,                    
        projector: StructuredProjector,
        device:    str | torch.device = "mps",
    ):
        self._llm       = llm
        self._text_enc  = text_enc
        self._proj      = projector
        self._device    = torch.device(device)
        self._proj.to(self._device)
        self._last_intent: str              = ""
        self._last_output: NavigatorOutput  = NavigatorOutput.identity()

    @torch.no_grad()
    def step(
        self,
        command:    str,
        speed_mps:  float = 0.0,
        roll_deg:   float = 0.0,
        pitch_deg:  float = 0.0,
        yaw_deg:    float = 0.0,
        altitude_m: float = 1.5,
    ) -> tuple[torch.Tensor, NavigatorOutput]:
        

        t0 = time.perf_counter()

        
        prompt = build_telemetry_prompt(
            command     = command,
            speed_mps   = speed_mps,
            roll_deg    = roll_deg,
            pitch_deg   = pitch_deg,
            yaw_deg     = yaw_deg,
            altitude_m  = altitude_m,
            last_intent = self._last_intent,
        )

        
        raw  = self._llm.reason(prompt)
        d    = _extract_json(raw)
        nav  = NavigatorOutput.from_dict(d) if d else NavigatorOutput.identity()
        self._last_intent = nav.intent
        self._last_output = nav

        
        enriched = nav.to_enriched_string()
        text_emb = self._text_enc(enriched)          
        text_emb = text_emb.squeeze(1).to(self._device)   

        
        num_vec = nav.to_vector().unsqueeze(0).to(self._device)   

        
        z_text = self._proj(text_emb, num_vec)       

        elapsed = time.perf_counter() - t0
        if elapsed > 0.9:
            print(f"[ReasoningNavigator] slow step: {elapsed*1000:.0f} ms")

        return z_text, nav

    @property
    def last_output(self) -> NavigatorOutput:
        
        return self._last_output

def build_reasoning_navigator(
    model_id:      str = "Qwen/Qwen2.5-3B-Instruct",
    device:        str = "mps",
    base_text_dim: int = 384,
    out_dim:       int = 384,
) -> tuple[ReasoningNavigator, StructuredProjector]:
    

    from nightly.vla_heads import TextEncoder

    llm       = ReasoningLLM(model_id=model_id, device=device)
    text_enc  = TextEncoder(device=device)
    projector = StructuredProjector(base_text_dim=base_text_dim, out_dim=out_dim)
    navigator = ReasoningNavigator(llm, text_enc, projector, device=device)
    return navigator, projector
