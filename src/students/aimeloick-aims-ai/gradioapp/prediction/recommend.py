import numpy as np


def compute_dummy_user(
        V,
        item_biases,
        dummy_ratings,
        K,
        lambda_val,
        tau,
        gamma_bias
    ):
    dummy_U = np.zeros(K, dtype=np.float32)
    dummy_bu = 0.0

    I_f = np.eye(K, dtype=np.float32)
    A = np.zeros((K, K), dtype=np.float32)
    b = np.zeros(K, dtype=np.float32)

    for movie_idx, r in dummy_ratings:
        v = V[movie_idx]
        bi = item_biases[movie_idx]
        A += np.outer(v, v)
        b += v * (r - bi - dummy_bu)

    try:
        dummy_U = np.linalg.solve(lambda_val * A + tau * I_f, lambda_val * b)
    except np.linalg.LinAlgError:
        print("⚠ Singular matrix — fallback to zeros for dummy_U")
        dummy_U = np.zeros(K, dtype=np.float32)


    residual_sum = 0.0  # noqa: E303
    for movie_idx, r in dummy_ratings:
        v = V[movie_idx]
        bi = item_biases[movie_idx]
        residual_sum += (r - np.dot(dummy_U, v) - bi)

    denom = (lambda_val * len(dummy_ratings) + gamma_bias)
    if denom != 0.0:
        dummy_bu = (lambda_val * residual_sum) / denom
    else:
        dummy_bu = 0.0

    return dummy_U.astype(np.float32), np.float32(dummy_bu)


def compute_overlap_stats(target_movie_idx, data_by_movie_train, dummy_movie_idx):
    # Find users who rated the LOTR movie (dummy_movie_idx)
    # Using data_by_movie_train for consistency with the training data.
    lotr_user_indices = {user_idx for user_idx, _ in data_by_movie_train[dummy_movie_idx]}

    # Find users who rated the target_movie_idx
    target_movie_user_indices = {user_idx for user_idx, _ in data_by_movie_train[target_movie_idx]}

    if not target_movie_user_indices:
        return {"prop_lotr_overlap": 0.0, "num_target_movie_raters": 0}

    # Calculate overlap
    overlapping_users = lotr_user_indices.intersection(target_movie_user_indices)
    prop_lotr_overlap = len(overlapping_users) / len(target_movie_user_indices)

    return {
        "prop_lotr_overlap": prop_lotr_overlap,
        "num_target_movie_raters": len(target_movie_user_indices)
    }


def predict_dummy_user(
    dummy_user_factors,
    V,
    item_biases,
    bias_scale=0.05
):

    num_items = V.shape[0]
    predictions_full_bias = []
    predictions_scaled_bias = []

    for item_idx in range(num_items):
        score_full_bias = (
            np.dot(dummy_user_factors, V[item_idx])
            + item_biases[item_idx]
        )

        score_scaled_bias = (
            np.dot(dummy_user_factors, V[item_idx])
            + bias_scale * item_biases[item_idx]
        )

        predictions_full_bias.append((item_idx, score_full_bias))
        predictions_scaled_bias.append((item_idx, score_scaled_bias))

    # Tri décroissant
    predictions_full_bias.sort(key=lambda x: x[1], reverse=True)
    predictions_scaled_bias.sort(key=lambda x: x[1], reverse=True)

    return predictions_full_bias, predictions_scaled_bias
