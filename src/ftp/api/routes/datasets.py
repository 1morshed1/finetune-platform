"""Dataset endpoints: upload + validate, list, fetch.

Upload runs the full validation pipeline **synchronously** and returns the quality
report in the response. That's a deliberate UX choice: the user should see whether
their data is good *immediately*, in the same request that uploaded it — the
report is the point of uploading.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from ftp.config import get_settings
from ftp.data.validator import validate_file
from ftp.db import Dataset, DatasetStatus, get_db, new_id

from ..serializers import dataset_dict

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])


@router.post("")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    db: Session = Depends(get_db),
):
    settings = get_settings()
    settings.ensure_dirs()

    ds_id = new_id("ds")
    dest = settings.datasets_dir / f"{ds_id}.jsonl"
    # Stream the upload to disk (don't hold the whole file in memory).
    content = await file.read()
    dest.write_bytes(content)

    ds = Dataset(
        id=ds_id,
        name=name or (file.filename or "dataset"),
        filename=file.filename or f"{ds_id}.jsonl",
        path=str(dest),
        status=DatasetStatus.VALIDATING,
    )
    db.add(ds)
    db.flush()

    # Run the validation pipeline now.
    result = validate_file(dest, settings)
    ds.num_examples = result.report.total_examples
    ds.quality_report = result.report.to_dict()
    ds.status = DatasetStatus.READY if result.status == "ready" else DatasetStatus.REJECTED
    if result.status != "ready":
        ds.error = "Validation failed; see quality_report.format_errors"
    db.flush()

    return dataset_dict(ds)


@router.get("")
async def list_datasets(db: Session = Depends(get_db)):
    rows = db.scalars(select(Dataset).order_by(Dataset.created_at.desc())).all()
    return [dataset_dict(d, include_report=False) for d in rows]


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, db: Session = Depends(get_db)):
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset_dict(ds)
