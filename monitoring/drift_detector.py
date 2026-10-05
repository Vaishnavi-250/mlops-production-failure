import numpy as np
from sklearn.datasets import load_breast_cancer


# ==========================================
# LOAD REFERENCE DATA
# ==========================================

data = load_breast_cancer()

reference_data = data.data


# ==========================================
# CREATE NEW DATA
# ==========================================

# Simulated production data
# Adding a shift to create a drift scenario

new_data = reference_data.copy()

new_data[:, 0] = new_data[:, 0] + 5


# ==========================================
# DRIFT DETECTION
# ==========================================

print("==========================================")
print("DATA DRIFT DETECTION")
print("==========================================")


drift_count = 0

threshold = 0.20


for feature_index in range(
    reference_data.shape[1]
):

    reference_mean = np.mean(
        reference_data[:, feature_index]
    )

    new_mean = np.mean(
        new_data[:, feature_index]
    )

    if reference_mean != 0:

        change = abs(
            new_mean - reference_mean
        ) / abs(reference_mean)

    else:

        change = 0


    if change > threshold:

        print(
            f"Feature {feature_index + 1}: "
            f"DRIFT DETECTED "
            f"({change * 100:.2f}% change)"
        )

        drift_count += 1

    else:

        print(
            f"Feature {feature_index + 1}: "
            f"Normal "
            f"({change * 100:.2f}% change)"
        )


# ==========================================
# FINAL RESULT
# ==========================================

print("\n==========================================")

if drift_count > 0:

    print("ALERT: DATA DRIFT DETECTED")

    print(
        "Drifted features:",
        drift_count
    )

else:

    print("No significant data drift detected.")


print("==========================================")