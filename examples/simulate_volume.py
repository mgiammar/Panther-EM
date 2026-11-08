"""Helper module for simulating cryo-EM volumes."""

import numpy as np
from ttsim3d.models import Simulator, SimulatorConfig


def simulate_volume(
    pdb_path: str,
    pixel_spacing: float = 0.4,
    volume_shape: tuple[int, int, int] = (256, 256, 256),
    voltage: float = 300.0,
    dose_start: float = 0.0,
    dose_end: float = 15.0,
    b_factor_scaling: float = 1.0,
    additional_b_factor: float = 30.0,
) -> np.ndarray:
    """Thin wrapper around ttsim3d objects to simulate a volume from a PDB/CIF file.

    Parameters
    ----------
    pdb_path : str
        Path to PDB/CIF file.
    pixel_spacing : float, optional
        Pixel spacing in Angstroms, by default 0.4.
    volume_shape : tuple[int, int, int], optional
        Shape of the output volume, by default (256, 256, 256).
    voltage : float, optional
        Microscope voltage in keV, by default 300.0.
    dose_start : float, optional
        Dose start in e-/A^2, by default 0.0.
    dose_end : float, optional
        Dose end in e-/A^2, by default 15.0.
    b_factor_scaling : float, optional
        B-factor scaling, by default 1.0.
    additional_b_factor : float, optional
        Additional B-factor, by default 30.0.

    Returns
    -------
    np.ndarray
        Simulated volume as numpy array.
    """
    sim_conf = SimulatorConfig(
        voltage=voltage,
        apply_dose_weighting=True,
        dose_start=dose_start,
        dose_end=dose_end,
        upsampling=-1,
    )

    sim = Simulator(
        pdb_filepath=pdb_path,
        pixel_spacing=pixel_spacing,
        volume_shape=volume_shape,
        b_factor_scaling=b_factor_scaling,
        additional_b_factor=additional_b_factor,
        simulator_config=sim_conf,
    )

    volume = sim.run()
    return volume.numpy()
