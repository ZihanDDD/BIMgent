"""Centralised LLM provider for the BIM-GUI agent pipeline.

Every LLM call in the project goes through this module.  To switch models
for an experiment, change the class-level ``MODEL_*`` attributes — all
downstream consumers pick up the change automatically.

Example
-------
>>> from BIMgent.provider.loop_providers.llm_provider import LLMProvider
>>> llm = LLMProvider()
>>> # text-only call
>>> text = llm.call(LLMProvider.MODEL_PLANNING, [prompt], top_p=0.95)
>>> # vision call
>>> img = LLMProvider.read_image("screenshot.png")
>>> text = llm.call(LLMProvider.MODEL_VISION, [prompt, img])
"""

import os
from google.genai import types
from BIMgent.utils.gemini_utils import gemini_call_with_retry, get_gemini_key_manager


class LLMProvider:
    """Single entry-point for all generative LLM calls.

    Model names are stored as **class attributes** so they can be overridden
    globally (``LLMProvider.MODEL_PLANNING = "new-model"``) or per-instance.
    """

    # ------------------------------------------------------------------
    # Model registry — change these to swap models across the pipeline
    # ------------------------------------------------------------------
    MODEL_PLANNING      = "gemini-2.5-pro"            # high / low-level planning
    MODEL_VISION        = "gemini-2.5-flash"           # action generation & supervision
    MODEL_UNDERSTANDING = "gemini-3.1-pro-preview"     # floorplan interpretation

    def __init__(self):
        self.key_mgr = get_gemini_key_manager()

    # ------------------------------------------------------------------
    # Core call
    # ------------------------------------------------------------------
    def call(
        self,
        model: str,
        contents: list,
        *,
        temperature: float = 0,
        top_p: float | None = None,
        thinking_budget: int | None = None,
        max_output_tokens: int | None = None,
    ) -> str | None:
        """Make an LLM call and return the response text.

        Parameters
        ----------
        model : str
            One of the ``MODEL_*`` class attributes, or any valid model id.
        contents : list
            Ordered list of content parts — plain strings and/or image
            ``Part`` objects returned by :meth:`read_image`.
        temperature : float
            Sampling temperature (default 0 = deterministic).
        top_p : float, optional
            Nucleus-sampling threshold.
        thinking_budget : int, optional
            Token budget for Gemini's extended-thinking mode.
        max_output_tokens : int, optional
            Hard cap on generated tokens.

        Returns
        -------
        str or None
            The model's text response, or *None* on failure.
        """
        config_kwargs: dict = {
            "response_modalities": ["Text"],
            "temperature": temperature,
        }
        if top_p is not None:
            config_kwargs["top_p"] = top_p
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        if max_output_tokens is not None:
            config_kwargs["max_output_tokens"] = max_output_tokens

        response = gemini_call_with_retry(
            self.key_mgr.client,
            model,
            contents,
            types.GenerateContentConfig(**config_kwargs),
        )

        if response is None:
            print(f"LLM call failed (model={model})")
            return None

        print(f"Response received from {model}")
        return response.text

    # ------------------------------------------------------------------
    # Image helper
    # ------------------------------------------------------------------
    @staticmethod
    def read_image(image_path: str) -> types.Part:
        """Read an image file and return a Gemini-compatible ``Part``."""
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        return types.Part.from_bytes(data=image_bytes, mime_type=mime)
