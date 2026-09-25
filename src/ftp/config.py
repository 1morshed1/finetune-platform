"""Application settings.

Everything configurable lives here and is driven by environment variables with
an ``FTP_`` prefix (e.g. ``FTP_TRAINER_BACKEND=hf``). We use pydantic-settings so
the config is typed and validated at startup rather than failing deep inside a
request.

Why a single settings object instead of reading os.environ everywhere?
- One place to see every knob the platform has.
- Types + defaults are enforced once.
- Tests can construct a Settings() with overrides instead of mutating global env.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FTP_", env_file=".env", extra="ignore")

    # --- Storage ----------------------------------------------------------
    # Root directory for all runtime state: the SQLite DB, uploaded datasets,
    # and model artifacts. Kept out of git (see .gitignore).
    data_dir: Path = Field(default=Path("data_store"))

    # --- Trainer ----------------------------------------------------------
    # Which trainer backend to use. "sim" = simulated (no GPU, default).
    # "hf" = real LoRA fine-tune via transformers+peft (needs the [ml] extra).
    # See ADR 0003.
    trainer_backend: str = Field(default="sim")

    # Deterministic seed so simulated runs (and shuffling/splitting) reproduce.
    seed: int = Field(default=42)

    # --- Deployment -------------------------------------------------------
    # "local" = load model into this process. "k8s" = render/apply manifests
    # (documented, not required locally). See ADR 0004.
    deploy_backend: str = Field(default="local")

    # --- Validation thresholds -------------------------------------------
    # These power the data-quality report's recommendations. Exposed as config
    # because "what counts as too small / too duplicated" is a policy decision a
    # platform operator should be able to tune without touching code.
    min_examples: int = Field(default=100)
    max_examples_warn: int = Field(default=10_000)
    max_duplicate_ratio: float = Field(default=0.10)  # >10% dupes -> warn
    max_low_quality_ratio: float = Field(default=0.10)
    near_duplicate_threshold: float = Field(default=0.85)  # Jaccard sim for MinHash

    # --- Cost model (USD) -------------------------------------------------
    # Rough public-ish rates used only for *estimates* in the quality report.
    # Documented as estimates in the UI; not billing.
    openai_ft_usd_per_1m_tokens: float = Field(default=8.0)
    local_gpu_usd_per_hour: float = Field(default=0.50)  # spot price ballpark

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ftp.db"

    @property
    def datasets_dir(self) -> Path:
        return self.data_dir / "datasets"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    def ensure_dirs(self) -> None:
        """Create the storage directories if they don't exist yet."""
        for d in (self.data_dir, self.datasets_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. Import this, don't construct Settings() directly, so the
    whole app shares one config (and one cache invalidation point in tests)."""
    return Settings()
