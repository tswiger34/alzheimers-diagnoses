from pydantic import BaseModel, ConfigDict, Field


class NiFTiMetadata(BaseModel):
    """Metadata emitted by dcm2niix for a converted NIfTI image.

    Attributes use Python snake_case names while Pydantic aliases preserve the
    original dcm2niix JSON keys for validation and serialization.
    """

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    modality: str = Field(alias="Modality", description="DICOM imaging modality for the source series.")
    magnetic_field_strength: float = Field(
        alias="MagneticFieldStrength", description="Nominal magnetic field strength of the MR scanner in tesla."
    )
    imaging_frequency: float = Field(
        alias="ImagingFrequency", description="MR imaging frequency used for the acquisition in megahertz."
    )
    manufacturer: str = Field(alias="Manufacturer", description="Scanner manufacturer name.")
    manufacturers_model_name: str = Field(
        alias="ManufacturersModelName", description="Scanner model name reported in the DICOM metadata."
    )
    institution_name: str = Field(
        alias="InstitutionName", description="Institution or site where the scan was acquired."
    )
    device_serial_number: str = Field(
        alias="DeviceSerialNumber", description="Scanner device serial number or anonymized scanner identifier."
    )
    body_part: str = Field(alias="BodyPart", description="Anatomical body part examined.")
    patient_position: str = Field(alias="PatientPosition", description="Patient position during acquisition.")
    software_versions: str = Field(
        alias="SoftwareVersions", description="Scanner software version string reported by the source DICOM data."
    )
    mr_acquisition_type: str = Field(
        alias="MRAcquisitionType", description="MR acquisition dimensionality or type."
    )
    study_description: str = Field(alias="StudyDescription", description="DICOM study description.")
    series_description: str = Field(alias="SeriesDescription", description="DICOM series description.")
    protocol_name: str = Field(alias="ProtocolName", description="Acquisition protocol name.")
    scanning_sequence: str = Field(alias="ScanningSequence", description="DICOM scanning sequence codes.")
    sequence_variant: str = Field(alias="SequenceVariant", description="DICOM sequence variant codes.")
    scan_options: str = Field(alias="ScanOptions", description="DICOM scan option codes.")
    sequence_name: str = Field(alias="SequenceName", description="Vendor-specific pulse sequence name.")
    image_type: list[str] = Field(
        alias="ImageType", description="DICOM image type components for the converted image."
    )
    nonlinear_gradient_correction: bool = Field(
        alias="NonlinearGradientCorrection",
        description="Whether nonlinear gradient correction was applied before or during conversion.",
    )
    series_number: int = Field(alias="SeriesNumber", description="DICOM series number.")
    acquisition_time: str = Field(
        alias="AcquisitionTime", description="Acquisition time from the source DICOM metadata."
    )
    acquisition_number: int = Field(alias="AcquisitionNumber", description="DICOM acquisition number.")
    slice_thickness: float = Field(alias="SliceThickness", description="Nominal slice thickness in millimeters.")
    sar: float = Field(alias="SAR", description="Specific absorption rate estimate reported for the acquisition.")
    table_position: list[float] = Field(alias="TablePosition", description="Scanner table position coordinates.")
    echo_time: float = Field(alias="EchoTime", description="Echo time in seconds.")
    repetition_time: float = Field(alias="RepetitionTime", description="Repetition time in seconds.")
    spoiling_state: bool = Field(alias="SpoilingState", description="Whether RF or gradient spoiling was enabled.")
    inversion_time: float = Field(alias="InversionTime", description="Inversion time in seconds.")
    flip_angle: float = Field(alias="FlipAngle", description="Nominal excitation flip angle in degrees.")
    partial_fourier: float = Field(alias="PartialFourier", description="Partial Fourier acquisition factor.")
    base_resolution: int = Field(alias="BaseResolution", description="Base acquisition matrix resolution.")
    shim_setting: list[int] = Field(
        alias="ShimSetting", description="Scanner shim settings recorded for the acquisition."
    )
    tx_ref_amp: float = Field(
        alias="TxRefAmp", description="Transmit reference amplitude reported by the scanner."
    )
    phase_resolution: float = Field(alias="PhaseResolution", description="Phase encoding resolution factor.")
    receive_coil_name: str = Field(alias="ReceiveCoilName", description="Receive coil name used for acquisition.")
    receive_coil_active_elements: str = Field(
        alias="ReceiveCoilActiveElements", description="Active receive coil elements used for acquisition."
    )
    pulse_sequence_details: str = Field(
        alias="PulseSequenceDetails", description="Vendor-specific pulse sequence detail string."
    )
    ref_lines_pe: int = Field(alias="RefLinesPE", description="Number of phase-encoding reference lines.")
    coil_combination_method: str = Field(
        alias="CoilCombinationMethod", description="Method used to combine receive coil data."
    )
    consistency_info: str = Field(
        alias="ConsistencyInfo", description="Vendor-specific sequence consistency or build information."
    )
    matrix_coil_mode: str = Field(alias="MatrixCoilMode", description="Coil or parallel imaging matrix mode.")
    percent_phase_fov: float = Field(alias="PercentPhaseFOV", description="Phase field-of-view percentage.")
    percent_sampling: float = Field(alias="PercentSampling", description="Frequency sampling percentage.")
    phase_encoding_steps: int = Field(alias="PhaseEncodingSteps", description="Number of phase encoding steps.")
    acquisition_matrix_pe: int = Field(
        alias="AcquisitionMatrixPE", description="Phase-encoding dimension of the acquisition matrix."
    )
    recon_matrix_pe: int = Field(
        alias="ReconMatrixPE", description="Phase-encoding dimension of the reconstructed matrix."
    )
    parallel_reduction_factor_in_plane: float = Field(
        alias="ParallelReductionFactorInPlane", description="In-plane acceleration factor for parallel imaging."
    )
    pixel_bandwidth: float = Field(alias="PixelBandwidth", description="Pixel bandwidth in hertz per pixel.")
    dwell_time: float = Field(alias="DwellTime", description="Readout dwell time in seconds.")
    image_orientation_patient_dicom: list[float] = Field(
        alias="ImageOrientationPatientDICOM", description="DICOM image orientation patient direction cosines."
    )
    in_plane_phase_encoding_direction_dicom: str = Field(
        alias="InPlanePhaseEncodingDirectionDICOM", description="DICOM in-plane phase encoding direction."
    )
    bids_guess: list[str] = Field(
        alias="BidsGuess", description="dcm2niix inferred BIDS datatype and filename suffix."
    )
    conversion_software: str = Field(
        alias="ConversionSoftware", description="Software used to convert the DICOM data."
    )
    conversion_software_version: str = Field(
        alias="ConversionSoftwareVersion", description="Version of the conversion software."
    )
