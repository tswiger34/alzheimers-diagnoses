import asyncio
import json
import zipfile
from pathlib import Path

import pytest
from sqlalchemy.dialects import postgresql

from mstat_project.data_prep import to_nifti


def _metadata_payload() -> dict[str, object]:
    return {
        "Modality": "MR",
        "MagneticFieldStrength": 3.0,
        "ImagingFrequency": 123.45,
        "Manufacturer": "SIEMENS",
        "ManufacturersModelName": "Prisma",
        "InstitutionName": "ADNI",
        "DeviceSerialNumber": "12345",
        "BodyPart": "BRAIN",
        "PatientPosition": "HFS",
        "SoftwareVersions": "syngo",
        "MRAcquisitionType": "3D",
        "StudyDescription": "ADNI",
        "SeriesDescription": "MPRAGE",
        "ProtocolName": "MPRAGE",
        "ScanningSequence": "GR",
        "SequenceVariant": "SP",
        "ScanOptions": "IR",
        "SequenceName": "tfl3d1",
        "ImageType": ["ORIGINAL", "PRIMARY", "M"],
        "NonlinearGradientCorrection": True,
        "SeriesNumber": 7,
        "AcquisitionTime": "120000.000000",
        "AcquisitionNumber": 1,
        "SliceThickness": 1.0,
        "SAR": 0.1,
        "TablePosition": [0.0, 1.0, 2.0],
        "EchoTime": 0.003,
        "RepetitionTime": 2.3,
        "SpoilingState": True,
        "InversionTime": 0.9,
        "FlipAngle": 9.0,
        "PartialFourier": 1.0,
        "BaseResolution": 256,
        "ShimSetting": [1, 2, 3],
        "TxRefAmp": 250.0,
        "PhaseResolution": 1.0,
        "ReceiveCoilName": "HeadNeck",
        "ReceiveCoilActiveElements": "HEA",
        "PulseSequenceDetails": "details",
        "RefLinesPE": 24,
        "CoilCombinationMethod": "Adaptive",
        "ConsistencyInfo": "N4",
        "MatrixCoilMode": "GRAPPA",
        "PercentPhaseFOV": 100.0,
        "PercentSampling": 100.0,
        "PhaseEncodingSteps": 240,
        "AcquisitionMatrixPE": 256,
        "ReconMatrixPE": 256,
        "ParallelReductionFactorInPlane": 2.0,
        "PixelBandwidth": 240.0,
        "DwellTime": 2.1e-06,
        "ImageOrientationPatientDICOM": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "InPlanePhaseEncodingDirectionDICOM": "COL",
        "BidsGuess": ["anat", "_T1w"],
        "ConversionSoftware": "dcm2niix",
        "ConversionSoftwareVersion": "v1",
    }


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


def test_load_json_metadata_validates_dcm2niix_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "scan.json"
    metadata_path.write_text(json.dumps(_metadata_payload()), encoding="utf-8")

    metadata = asyncio.run(to_nifti.load_json_metadata(metadata_path))

    assert metadata.modality == "MR"
    assert metadata.magnetic_field_strength == 3.0
    assert metadata.image_type == ["ORIGINAL", "PRIMARY", "M"]


def test_load_cohort_metadata_extracts_records_and_replaces_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nifti_dir = tmp_path / "nifti"
    nifti_dir.mkdir()
    with zipfile.ZipFile(nifti_dir / "cohort_01.zip", mode="w") as zf:
        zf.writestr("101/scan.json", json.dumps(_metadata_payload()))
        zf.writestr("101/scan.nii.gz", b"nifti")

    calls: list[tuple[list[dict[str, object]], str]] = []

    def fake_replace(records: list[dict[str, object]], cohort_str: str) -> None:
        calls.append((records, cohort_str))

    monkeypatch.setattr(to_nifti, "NIFTI_IMAGE_PATH", nifti_dir)
    monkeypatch.setattr(to_nifti, "_replace_cohort_metadata", fake_replace)

    asyncio.run(to_nifti.load_cohort_metadata(1))

    assert len(calls) == 1
    records, cohort_str = calls[0]
    assert cohort_str == "01"
    assert len(records) == 1
    assert records[0]["image_id"] == "101"
    assert records[0]["cohort"] == "01"
    assert records[0]["source_json_path"] == "101/scan.json"
    assert records[0]["modality"] == "MR"
    assert records[0]["image_type"] == ["ORIGINAL", "PRIMARY", "M"]


def test_delete_cohort_metadata_filters_to_requested_cohort() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.statements: list[object] = []

        def execute(self, statement: object, *_args: object, **_kwargs: object) -> None:
            self.statements.append(statement)

    conn = FakeConnection()

    to_nifti._delete_cohort_metadata(conn, "07")  # type: ignore[arg-type]

    assert len(conn.statements) == 1
    compiled = str(
        conn.statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "DELETE FROM raw.raw_nifti_metadata" in compiled
    assert "raw.raw_nifti_metadata.cohort = '07'" in compiled
