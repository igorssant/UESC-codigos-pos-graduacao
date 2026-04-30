import nibabel as nib
import numpy as np


def main() -> None:
    image: nib.nifti1.Nifti1Image  = nib.load("images/120.label.nii.gz")
    data: np.ndarray = image.get_fdata().astype(np.float64)
    flattened_data: np.ndarray = data.flatten()

    with open("input3.txt", "w") as file_descriptor:
        file_descriptor.write(f"{data.shape[0]}\n{data.shape[1]}\n{data.shape[2]}\n")
        np.savetxt(file_descriptor, [flattened_data], fmt="%lf", delimiter=" ")

if __name__ == "__main__":
    main()

