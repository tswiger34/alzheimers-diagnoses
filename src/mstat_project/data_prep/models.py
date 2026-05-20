from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class NiFTiMetadata(BaseModel):
    """Metadata emitted by dcm2niix for a converted NIfTI image.

    Attributes use Python snake_case names while Pydantic aliases preserve the
    original dcm2niix JSON keys for validation and serialization.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    modality: Optional[str] = Field(
        default=None, alias="Modality", description="DICOM imaging modality for the source series."
    )
    magnetic_field_strength: Optional[float] = Field(
        default=None,
        alias="MagneticFieldStrength",
        description="Nominal magnetic field strength of the MR scanner in tesla.",
    )
    imaging_frequency: Optional[float] = Field(
        default=None,
        alias="ImagingFrequency",
        description="MR imaging frequency used for the acquisition in megahertz.",
    )
    manufacturer: Optional[str] = Field(
        default=None, alias="Manufacturer", description="Scanner manufacturer name."
    )
    manufacturers_model_name: Optional[str] = Field(
        default=None,
        alias="ManufacturersModelName",
        description="Scanner model name reported in the DICOM metadata.",
    )
    institution_name: Optional[str] = Field(
        default=None, alias="InstitutionName", description="Institution or site where the scan was acquired."
    )
    device_serial_number: Optional[str] = Field(
        default=None,
        alias="DeviceSerialNumber",
        description="Scanner device serial number or anonymized scanner identifier.",
    )
    body_part: Optional[str] = Field(default=None, alias="BodyPart", description="Anatomical body part examined.")
    patient_position: Optional[str] = Field(
        default=None, alias="PatientPosition", description="Patient position during acquisition."
    )
    software_versions: Optional[str] = Field(
        default=None,
        alias="SoftwareVersions",
        description="Scanner software version string reported by the source DICOM data.",
    )
    mr_acquisition_type: Optional[str] = Field(
        default=None, alias="MRAcquisitionType", description="MR acquisition dimensionality or type."
    )
    study_description: Optional[str] = Field(
        default=None, alias="StudyDescription", description="DICOM study description."
    )
    series_description: Optional[str] = Field(
        default=None, alias="SeriesDescription", description="DICOM series description."
    )
    protocol_name: Optional[str] = Field(
        default=None, alias="ProtocolName", description="Acquisition protocol name."
    )
    scanning_sequence: Optional[str] = Field(
        default=None, alias="ScanningSequence", description="DICOM scanning sequence codes."
    )
    sequence_variant: Optional[str] = Field(
        default=None, alias="SequenceVariant", description="DICOM sequence variant codes."
    )
    scan_options: Optional[str] = Field(default=None, alias="ScanOptions", description="DICOM scan option codes.")
    sequence_name: Optional[str] = Field(
        default=None, alias="SequenceName", description="Vendor-specific pulse sequence name."
    )
    image_type: Optional[list[str]] = Field(
        default=None, alias="ImageType", description="DICOM image type components for the converted image."
    )
    nonlinear_gradient_correction: Optional[bool] = Field(
        default=None,
        alias="NonlinearGradientCorrection",
        description="Whether nonlinear gradient correction was applied before or during conversion.",
    )
    series_number: Optional[int] = Field(default=None, alias="SeriesNumber", description="DICOM series number.")
    acquisition_time: Optional[str] = Field(
        default=None, alias="AcquisitionTime", description="Acquisition time from the source DICOM metadata."
    )
    acquisition_number: Optional[int] = Field(
        default=None, alias="AcquisitionNumber", description="DICOM acquisition number."
    )
    slice_thickness: Optional[float] = Field(
        default=None, alias="SliceThickness", description="Nominal slice thickness in millimeters."
    )
    sar: Optional[float] = Field(
        default=None, alias="SAR", description="Specific absorption rate estimate reported for the acquisition."
    )
    table_position: Optional[list[float]] = Field(
        default=None, alias="TablePosition", description="Scanner table position coordinates."
    )
    echo_time: Optional[float] = Field(default=None, alias="EchoTime", description="Echo time in seconds.")
    repetition_time: Optional[float] = Field(
        default=None, alias="RepetitionTime", description="Repetition time in seconds."
    )
    spoiling_state: Optional[bool] = Field(
        default=None, alias="SpoilingState", description="Whether RF or gradient spoiling was enabled."
    )
    inversion_time: Optional[float] = Field(
        default=None, alias="InversionTime", description="Inversion time in seconds."
    )
    flip_angle: Optional[float] = Field(
        default=None, alias="FlipAngle", description="Nominal excitation flip angle in degrees."
    )
    partial_fourier: Optional[float] = Field(
        default=None, alias="PartialFourier", description="Partial Fourier acquisition factor."
    )
    base_resolution: Optional[int] = Field(
        default=None, alias="BaseResolution", description="Base acquisition matrix resolution."
    )
    shim_setting: Optional[list[int]] = Field(
        default=None, alias="ShimSetting", description="Scanner shim settings recorded for the acquisition."
    )
    tx_ref_amp: Optional[float] = Field(
        default=None, alias="TxRefAmp", description="Transmit reference amplitude reported by the scanner."
    )
    phase_resolution: Optional[float] = Field(
        default=None, alias="PhaseResolution", description="Phase encoding resolution factor."
    )
    receive_coil_name: Optional[str] = Field(
        default=None, alias="ReceiveCoilName", description="Receive coil name used for acquisition."
    )
    receive_coil_active_elements: Optional[str] = Field(
        default=None,
        alias="ReceiveCoilActiveElements",
        description="Active receive coil elements used for acquisition.",
    )
    pulse_sequence_details: Optional[str] = Field(
        default=None, alias="PulseSequenceDetails", description="Vendor-specific pulse sequence detail string."
    )
    ref_lines_pe: Optional[int] = Field(
        default=None, alias="RefLinesPE", description="Number of phase-encoding reference lines."
    )
    coil_combination_method: Optional[str] = Field(
        default=None, alias="CoilCombinationMethod", description="Method used to combine receive coil data."
    )
    consistency_info: Optional[str] = Field(
        default=None,
        alias="ConsistencyInfo",
        description="Vendor-specific sequence consistency or build information.",
    )
    matrix_coil_mode: Optional[str] = Field(
        default=None, alias="MatrixCoilMode", description="Coil or parallel imaging matrix mode."
    )
    percent_phase_fov: Optional[float] = Field(
        default=None, alias="PercentPhaseFOV", description="Phase field-of-view percentage."
    )
    percent_sampling: Optional[float] = Field(
        default=None, alias="PercentSampling", description="Frequency sampling percentage."
    )
    phase_encoding_steps: Optional[int] = Field(
        default=None, alias="PhaseEncodingSteps", description="Number of phase encoding steps."
    )
    acquisition_matrix_pe: Optional[int] = Field(
        default=None,
        alias="AcquisitionMatrixPE",
        description="Phase-encoding dimension of the acquisition matrix.",
    )
    recon_matrix_pe: Optional[int] = Field(
        default=None, alias="ReconMatrixPE", description="Phase-encoding dimension of the reconstructed matrix."
    )
    parallel_reduction_factor_in_plane: Optional[float] = Field(
        default=None,
        alias="ParallelReductionFactorInPlane",
        description="In-plane acceleration factor for parallel imaging.",
    )
    pixel_bandwidth: Optional[float] = Field(
        default=None, alias="PixelBandwidth", description="Pixel bandwidth in hertz per pixel."
    )
    dwell_time: Optional[float] = Field(
        default=None, alias="DwellTime", description="Readout dwell time in seconds."
    )
    image_orientation_patient_dicom: Optional[list[float]] = Field(
        default=None,
        alias="ImageOrientationPatientDICOM",
        description="DICOM image orientation patient direction cosines.",
    )
    in_plane_phase_encoding_direction_dicom: Optional[str] = Field(
        default=None,
        alias="InPlanePhaseEncodingDirectionDICOM",
        description="DICOM in-plane phase encoding direction.",
    )
    bids_guess: Optional[list[str]] = Field(
        default=None, alias="BidsGuess", description="dcm2niix inferred BIDS datatype and filename suffix."
    )
    conversion_software: Optional[str] = Field(
        default=None, alias="ConversionSoftware", description="Software used to convert the DICOM data."
    )
    conversion_software_version: Optional[str] = Field(
        default=None, alias="ConversionSoftwareVersion", description="Version of the conversion software."
    )
