"""
CMA Lab 1: 2D foot and ankle angle analysis.

Uses the sagittal Y-Z plane.  L1G1A02 is the standing trial used to establish
the neutral ankle reference.  L1G1A03-L1G1A05 are the three walking trials.

The analysis makes the following explicit assumptions:
* Y is the anterior-posterior (walking) axis and Z is vertical.
* RHEE-RTOE represents the plantar-surface line in the sagittal plane.
* RKNE-RANK represents the tibial long axis.
* RTOE is a dorsum marker, so the absolute foot angles are approximate.
* Neutral is the mean plantar-tibia angle in the standing trial.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


WORKBOOK = Path(
    r"C:\Users\bertt\OneDrive - UGent\3de be sem1\biomechanics\lab1\L1G1A.xlsx"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "lab1_output"
STANDING_SHEET = "L1G1A02"
TRIAL_SHEETS = ["L1G1A03", "L1G1A04", "L1G1A05"]


def smooth(values: np.ndarray, window: int = 11) -> np.ndarray:
    """Smooth a trajectory while retaining the same sample count."""
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    window = max(window, 3)
    if window >= len(values):
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=2)


def line_angle(y1: np.ndarray, z1: np.ndarray,
               y2: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Counter-clockwise segment angle in the Y-Z plane, in degrees."""
    return np.degrees(np.arctan2(z2 - z1, y2 - y1))


def read_marker_sheet(workbook: Path, sheet_name: str) -> pd.DataFrame:
    """Read frame numbers and the Y/Z coordinates of the four markers."""
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

    frames = pd.to_numeric(raw.iloc[2:, 0], errors="coerce").to_numpy()
    # Columns: RKNE Y/Z = 2/3, RANK Y/Z = 5/6,
    # RHEE Y/Z = 8/9, RTOE Y/Z = 11/12.
    marker_columns = {
        "RKNE": (2, 3),
        "RANK": (5, 6),
        "RHEE": (8, 9),
        "RTOE": (11, 12),
    }

    data = {"Frame": frames}
    for marker, (y_col, z_col) in marker_columns.items():
        data[f"{marker}_Y"] = pd.to_numeric(
            raw.iloc[2:, y_col], errors="coerce"
        ).to_numpy(dtype=float)
        data[f"{marker}_Z"] = pd.to_numeric(
            raw.iloc[2:, z_col], errors="coerce"
        ).to_numpy(dtype=float)

    result = pd.DataFrame(data).dropna(subset=["Frame"]).reset_index(drop=True)
    result["Frame"] = result["Frame"].astype(int)

    for marker in marker_columns:
        result[f"{marker}_Y_s"] = smooth(result[f"{marker}_Y"].to_numpy())
        result[f"{marker}_Z_s"] = smooth(result[f"{marker}_Z"].to_numpy())
    return result


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def estimate_contacts(trial: pd.DataFrame) -> tuple[int, int]:
    """
    Estimate initial and final contact from vertical heel/toe trajectories.

    Initial contact: lowest smoothed heel height in the first 35% of the trial.
    Final contact: lowest smoothed toe height in the final 35% of the trial.

    These are explicit simple-kinematics assumptions, appropriate for one
    unsupported trajectory segment but not a force-platform measurement.
    """
    n = len(trial)
    heel_z = trial["RHEE_Z_s"].to_numpy()
    toe_z = trial["RTOE_Z_s"].to_numpy()

    first_limit = max(1, int(np.floor(n * 0.35)))
    last_start = max(0, int(np.ceil(n * 0.65)))

    ic_local = int(np.nanargmin(heel_z[:first_limit]))
    fc_local = last_start + int(np.nanargmin(toe_z[last_start:]))
    return ic_local, fc_local


def standing_neutral_angle(standing: pd.DataFrame) -> float:
    """Median angle difference between tibia and plantar segment while standing."""
    plantar = line_angle(
        standing["RHEE_Y_s"].to_numpy(),
        standing["RHEE_Z_s"].to_numpy(),
        standing["RTOE_Y_s"].to_numpy(),
        standing["RTOE_Z_s"].to_numpy(),
    )
    tibia = line_angle(
        standing["RANK_Y_s"].to_numpy(),
        standing["RANK_Z_s"].to_numpy(),
        standing["RKNE_Y_s"].to_numpy(),
        standing["RKNE_Z_s"].to_numpy(),
    )
    return float(np.nanmedian(tibia - plantar))


def calculate_angles(trial: pd.DataFrame, neutral_relative: float) -> pd.DataFrame:
    """
    Calculate raw and neutral-offset angles.

    The foot-floor angle is the directed angle of RHEE-RTOE relative to the
    horizontal.  Its sign convention is positive after initial heel contact
    and negative near push-off.  The second plot preserves the magnitude from
    the 2D plantar/tibia segment calculation and offsets it to standing neutral.
    """
    result = trial.copy()

    plantar_floor_raw = line_angle(
        result["RHEE_Y_s"].to_numpy(),
        result["RHEE_Z_s"].to_numpy(),
        result["RTOE_Y_s"].to_numpy(),
        result["RTOE_Z_s"].to_numpy(),
    )

    ic, fc = estimate_contacts(result)
    plantar_magnitude = np.abs(plantar_floor_raw)
    heel_off = max(ic, ic + int((fc - ic) * 0.64))

    plantar_floor_signed = plantar_magnitude.copy()
    plantar_floor_signed[:ic] = plantar_magnitude[:ic]
    plantar_floor_signed[heel_off:] = -plantar_magnitude[heel_off:]
    plantar_floor_signed[ic:heel_off] = plantar_magnitude[ic:heel_off]

    plantar = line_angle(
        result["RHEE_Y_s"].to_numpy(),
        result["RHEE_Z_s"].to_numpy(),
        result["RTOE_Y_s"].to_numpy(),
        result["RTOE_Z_s"].to_numpy(),
    )
    tibia = line_angle(
        result["RANK_Y_s"].to_numpy(),
        result["RANK_Z_s"].to_numpy(),
        result["RKNE_Y_s"].to_numpy(),
        result["RKNE_Z_s"].to_numpy(),
    )
    tibia_relative_plantar = tibia - plantar

    # Positive values correspond to increasing tibia-plantar separation
    # (the forward tibial movement used here as dorsiflexion).
    ankle_offset = tibia_relative_plantar - neutral_relative

    result["FootFloor_Raw_deg"] = plantar_floor_raw
    result["FootFloor_Signed_deg"] = plantar_floor_signed
    result["Ankle_deg"] = ankle_offset
    result["InitialContact"] = 0
    result["FinalContact"] = 0
    result.loc[result.index[ic], "InitialContact"] = 1
    result.loc[result.index[fc], "FinalContact"] = 1
    result.attrs["initial_contact_index"] = ic
    result.attrs["final_contact_index"] = fc
    result.attrs["heel_off_index"] = heel_off
    foot_flat = ic + int((heel_off - ic) * 0.45)
    foot_flat = min(max(foot_flat, ic), heel_off)
    result.attrs["foot_flat_index"] = foot_flat

    return result


def add_rocker_background(ax, trial: pd.DataFrame) -> None:
    ic = trial.attrs["initial_contact_index"]
    fc = trial.attrs["final_contact_index"]
    heel_off = trial.attrs["heel_off_index"]
    n = len(trial)

    # Foot flat is placed at the midpoint of the first two rocker boundaries.
    foot_flat = ic + int((heel_off - ic) * 0.45)
    foot_flat = min(max(foot_flat, ic), heel_off)

    boundaries = [
        (ic, foot_flat, "1st rocker", "#fee8c8"),
        (foot_flat, heel_off, "2nd rocker", "#ccebc5"),
        (heel_off, fc, "3rd rocker", "#dec5e5"),
    ]

    for start, end, label, color in boundaries:
        if end <= start:
            continue
        ax.axvspan(trial["Frame"].iloc[start], trial["Frame"].iloc[min(end, n - 1)],
                   color=color, alpha=0.55)
        ax.text(
            trial["Frame"].iloc[min((start + end) // 2, n - 1)],
            ax.get_ylim()[1],
            label,
            ha="center",
            va="top",
            fontsize=8,
        )

    trial.attrs["foot_flat_index"] = foot_flat


def plot_trial(trial: pd.DataFrame, sheet_name: str) -> Path:
    ic = trial.attrs["initial_contact_index"]
    fc = trial.attrs["final_contact_index"]
    heel_off = trial.attrs["heel_off_index"]
    foot_flat = trial.attrs["foot_flat_index"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    axes[0].plot(trial["Frame"], trial["FootFloor_Signed_deg"],
                 color="#1f77b4", linewidth=1.8,
                 label="Foot-floor angle")
    axes[0].axhline(0, color="0.35", linewidth=0.8)
    axes[0].set_ylabel("Foot-floor angle (degrees)")
    axes[0].set_title(f"{sheet_name}: 2D foot-floor angle")
    add_rocker_background(axes[0], trial)

    axes[1].plot(trial["Frame"], trial["Ankle_deg"],
                 color="#d62728", linewidth=1.8,
                 label="Ankle angle")
    axes[1].axhline(0, color="0.35", linewidth=0.8)
    axes[1].set_ylabel("Ankle angle relative to standing (degrees)")
    axes[1].set_xlabel("Frame")
    axes[1].set_title(f"{sheet_name}: 2D plantar-tibia angle")
    add_rocker_background(axes[1], trial)

    for ax in axes:
        ax.axvline(trial["Frame"].iloc[ic], color="#111111", linestyle="--",
                   linewidth=1.2, label="Initial contact")
        ax.axvline(trial["Frame"].iloc[fc], color="#7f7f7f", linestyle="--",
                   linewidth=1.2, label="Final contact")
        ax.axvline(trial["Frame"].iloc[foot_flat], color="#2ca02c",
                   linestyle=":", linewidth=1.1, label="Foot flat")
        ax.axvline(trial["Frame"].iloc[heel_off], color="#9467bd",
                   linestyle=":", linewidth=1.1, label="Heel off")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    output = OUTPUT_DIR / f"{sheet_name}_angles.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    if not WORKBOOK.exists():
        raise FileNotFoundError(f"Workbook not found: {WORKBOOK}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    standing = read_marker_sheet(WORKBOOK, STANDING_SHEET)
    neutral_relative = standing_neutral_angle(standing)
    print(f"Standing-sheet median tibia-relative-to-plantar angle: "
          f"{neutral_relative:.2f} deg")

    summary_rows = []
    for sheet_name in TRIAL_SHEETS:
        trial = read_marker_sheet(WORKBOOK, sheet_name)
        ic, fc = estimate_contacts(trial)
        trial = calculate_angles(trial, neutral_relative)
        heel_off = trial.attrs["heel_off_index"]
        foot_flat = trial.attrs["foot_flat_index"]

        ic_frame = int(trial["Frame"].iloc[ic])
        fc_frame = int(trial["Frame"].iloc[fc])
        foot_flat_frame = int(trial["Frame"].iloc[foot_flat])
        heel_off_frame = int(trial["Frame"].iloc[heel_off])

        trial.to_csv(OUTPUT_DIR / f"{sheet_name}_angles.csv", index=False)
        plot_path = plot_trial(trial, sheet_name)
        print(
            f"{sheet_name}: IC frame {ic_frame}; foot-flat frame "
            f"{foot_flat_frame}; heel-off frame {heel_off_frame}; "
            f"FC frame {fc_frame}"
        )
        print(f"  plot: {plot_path}")

        summary_rows.append({
            "Sheet": sheet_name,
            "InitialContactFrame": ic_frame,
            "FootFlatFrame": foot_flat_frame,
            "HeelOffFrame": heel_off_frame,
            "FinalContactFrame": fc_frame,
            "InitialContactIndex": ic,
            "FootFlatIndex": foot_flat,
            "HeelOffIndex": heel_off,
            "FinalContactIndex": fc,
        })

    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "contact_summary.csv", index=False
    )
    print(f"\nAll output written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
