import csv


def build_dummy_rating(
    movie_title,
    movieid_to_idx,
    rating_value=5.0,
    movies_csv_path="ml-32m/movies.csv"
):
    movieid_to_title = {}
    with open(movies_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            movieid_to_title[int(row[0])] = row[1]

    movie_id = None
    for mid, title in movieid_to_title.items():
        if title == movie_title:
            movie_id = mid
            break

    if movie_id is None:
        raise ValueError(f"Film non trouvé : {movie_title}")

    if movie_id not in movieid_to_idx:
        raise ValueError(f"movieId {movie_id} absent du mapping interne")

    dummy_movie_idx = movieid_to_idx[movie_id]

    dummy_ratings = [(dummy_movie_idx, float(rating_value))]

    return dummy_ratings, dummy_movie_idx, movie_id, movieid_to_title


def build_dummy_rating_from_indices(
    movie_idx_list,
    rating_list,
    movies_csv_path="ml-32m/movies.csv",
    movieid_to_idx=None
):
    if len(movie_idx_list) != len(rating_list):
        raise ValueError("Les listes d'indices et de ratings doivent avoir la même longueur")

    dummy_ratings = [(idx, float(r)) for idx, r in zip(movie_idx_list, rating_list)]
    dummy_movie_idx = movie_idx_list[0] if movie_idx_list else None

    # Construire movieid_to_title depuis CSV
    movieid_to_title = {}
    with open(movies_csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            movieid_to_title[int(row[0])] = row[1]

    # Si movieid_to_idx fourni, retrouver le premier movie_id pour retour
    movie_id = None
    if movieid_to_idx and dummy_movie_idx is not None:
        for mid, idx in movieid_to_idx.items():
            if idx == dummy_movie_idx:
                movie_id = mid
                break

    return dummy_ratings, dummy_movie_idx, movie_id, movieid_to_title
