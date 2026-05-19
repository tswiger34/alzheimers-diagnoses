import asyncio
import zipfile
from pathlib import Path

import pytest

from mstat_project.data_prep import to_nifti


def _write_cohort_zip(path: Path, image_ids: list[str]) -> None:
    with zipfile.ZipFile(path, mode="w") as zf:
        for image_id in image_ids:
            zf.writestr(f"ADNI/site/subject/visit/I{image_id}/scan_{image_id}.dcm", b"dicom")

        zf.writestr("ADNI/site/subject/visit/I99999/notes.txt", "not a dicom")


def test_resolve_max_workers_uses_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(to_nifti.NIFTI_WORKERS_ENV_VAR, "3")

    assert to_nifti._resolve_max_workers(None) == 3


def test_resolve_max_workers_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(to_nifti.NIFTI_WORKERS_ENV_VAR, "not-an-int")

    with pytest.raises(ValueError, match=to_nifti.NIFTI_WORKERS_ENV_VAR):
        to_nifti._resolve_max_workers(None)

    with pytest.raises(ValueError, match="at least 1"):
        to_nifti._resolve_max_workers(0)


def test_find_selected_dicom_members_groups_only_selected_dicoms(tmp_path: Path) -> None:
    zip_path = tmp_path / "cohort_01.zip"
    _write_cohort_zip(zip_path, ["101", "202"])

    members = to_nifti._find_selected_dicom_members(zip_path, {"202", "99999"})

    assert members == {"202": ["ADNI/site/subject/visit/I202/scan_202.dcm"]}


def test_process_cohort_async_limits_workers_and_reports_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_images_dir = tmp_path / "raw_images"
    raw_images_dir.mkdir()
    _write_cohort_zip(raw_images_dir / "cohort_01.zip", ["101", "202", "303", "404"])

    existing_output_dir = tmp_path / "data/images/nifti/cohort_01/202"
    existing_output_dir.mkdir(parents=True)
    (existing_output_dir / "existing.json").write_text("already converted")

    active_conversions = 0
    max_active_conversions = 0
    extracted_dirs: list[Path] = []

    def fake_extract_dicom_members(_zip_path: Path, _member_names: list[str], output_dir: Path) -> None:
        output_dir.mkdir(parents=True)
        (output_dir / "scan.dcm").write_bytes(b"dicom")
        extracted_dirs.append(output_dir)

    async def fake_dicom_to_nifti_async(dicom_dir: Path, output_dir: Path) -> None:
        nonlocal active_conversions, max_active_conversions

        assert dicom_dir.exists()
        active_conversions += 1
        max_active_conversions = max(max_active_conversions, active_conversions)
        try:
            await asyncio.sleep(0.01)
            if output_dir.name == "303":
                raise RuntimeError("conversion failed")
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "converted.nii.gz").write_bytes(b"nifti")
        finally:
            active_conversions -= 1

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("IMAGES_PATH", str(raw_images_dir))
    monkeypatch.setattr(to_nifti, "_extract_dicom_members", fake_extract_dicom_members)
    monkeypatch.setattr(to_nifti, "dicom_to_nifti_async", fake_dicom_to_nifti_async)

    processed_ids = asyncio.run(
        to_nifti.process_cohort_async(
            1,
            {"101", "202", "303", "404"},
            max_workers=2,
            archive=False,
        )
    )

    assert processed_ids == {"101", "202", "404"}
    assert max_active_conversions <= 2
    assert extracted_dirs
    assert all(not path.exists() for path in extracted_dirs)
    assert not (tmp_path / "data/images/nifti/cohort_01/303/converted.nii.gz").exists()
