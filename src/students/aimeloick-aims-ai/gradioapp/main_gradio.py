import json
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd
import requests
from explainer.contribution_items_passed import top_historical_contributions
from prediction.recommend import compute_dummy_user
from user_ratings_manager import UserRatingsManager

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset directory - assumes files are already present
DATA_DIR = "./ml-32m"

# TMDB Configuration
TMDB_API_KEY = "42a652236ac00d61b4fa5639e900a765"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_MOVIE_DETAILS_URL = "https://api.themoviedb.org/3/movie"

CACHE_FILE = "movie_images_cache.json"
DESCRIPTIONS_CACHE_FILE = "movie_descriptions_cache.json"
IMAGE_CACHE = {}
DESCRIPTIONS_CACHE = {}

# ============================================================================
# LOAD ALS MODEL
# ============================================================================

try:
    with open("als_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)

    user_factors = loaded_model["user_factors"]
    item_factors = loaded_model["item_factors"]
    user_biases = loaded_model["user_biases"]
    item_biases = loaded_model["item_biases"]
    movieid_to_idx = loaded_model["movieid_to_idx"]
    idx_to_movie = loaded_model["idx_to_movie"]
    print("✅ ALS model loaded successfully")
except Exception as e:
    print(f"⚠️ ALS model not found - recommendations will be limited: {e}")
    user_factors = item_factors = user_biases = item_biases = None
    movieid_to_idx = idx_to_movie = {}


# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

def load_image_cache():
    """Load image cache from JSON file."""
    global IMAGE_CACHE
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            IMAGE_CACHE = json.load(f)
        print(f"📦 {len(IMAGE_CACHE)} images loaded from cache")
    else:
        IMAGE_CACHE = {}


def save_image_cache():
    """Save image cache to JSON file."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(IMAGE_CACHE, f, ensure_ascii=False, indent=2)


def load_descriptions_cache():
    """Load descriptions cache from JSON file."""
    global DESCRIPTIONS_CACHE
    if os.path.exists(DESCRIPTIONS_CACHE_FILE):
        with open(DESCRIPTIONS_CACHE_FILE, encoding="utf-8") as f:
            DESCRIPTIONS_CACHE = json.load(f)
        print(f"📝 {len(DESCRIPTIONS_CACHE)} descriptions loaded from cache")
    else:
        DESCRIPTIONS_CACHE = {}


def save_descriptions_cache():
    """Save descriptions cache to JSON file."""
    with open(DESCRIPTIONS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(DESCRIPTIONS_CACHE, f, ensure_ascii=False, indent=2)


def get_tmdb_image_and_description(title, movie_id):
    """Fetch image and description from TMDB."""
    cache_key = str(movie_id)

    image_url = IMAGE_CACHE.get(cache_key)
    description = DESCRIPTIONS_CACHE.get(cache_key, "")

    if image_url and description:
        return image_url, description

    try:
        clean_title = title.split("(")[0].strip()

        params = {
            "api_key": TMDB_API_KEY,
            "query": clean_title,
            "language": "en-US",
        }
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=5)
        data = response.json()

        if data.get("results") and len(data["results"]) > 0:
            result = data["results"][0]
            tmdb_id = result.get("id")

            poster_path = result.get("poster_path")
            if poster_path and not image_url:
                image_url = f"{TMDB_IMAGE_BASE}{poster_path}"
                IMAGE_CACHE[cache_key] = image_url
                save_image_cache()

            if not description and tmdb_id:
                details_response = requests.get(
                    f"{TMDB_MOVIE_DETAILS_URL}/{tmdb_id}",
                    params={"api_key": TMDB_API_KEY, "language": "en-US"},
                    timeout=5,
                )
                details = details_response.json()
                description = details.get("overview", "")
                if description:
                    DESCRIPTIONS_CACHE[cache_key] = description
                    save_descriptions_cache()

        if not image_url:
            placeholder_url = (
                f"https://via.placeholder.com/500x750/1a1a2e/ff6b6b"
                f"?text={clean_title.replace(' ', '+')}"
            )
            image_url = placeholder_url
            IMAGE_CACHE[cache_key] = image_url

        return image_url, description

    except Exception:
        placeholder = (
            f"https://via.placeholder.com/500x750/1a1a2e/fff"
            f"?text={title.replace(' ', '+')}"
        )
        IMAGE_CACHE[cache_key] = placeholder
        return placeholder, ""


def get_tmdb_image(title, movie_id):
    """Fetch only image."""
    image, _ = get_tmdb_image_and_description(title, movie_id)
    return image


# ============================================================================
# DATA LOADING - SIMPLIFIED (NO DOWNLOAD)
# ============================================================================

def load_movielens_data(data_dir=DATA_DIR):
    """Load MovieLens data from CSV files (assumes files are already present)."""
    print("\n" + "=" * 60)
    print("📁 LOADING MOVIELENS DATA")
    print("=" * 60)

    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"❌ Data directory '{data_dir}' not found!\n"
            f"Please ensure the MovieLens dataset is extracted to '{data_dir}'"
        )

    # Load movies
    movies_path = os.path.join(data_dir, "movies.csv")
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"❌ File not found: {movies_path}")

    movies_df = pd.read_csv(movies_path, encoding="utf-8")
    print(f"✓ {len(movies_df)} movies loaded")

    # Load ratings
    ratings_path = os.path.join(data_dir, "ratings.csv")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"❌ File not found: {ratings_path}")

    print("Loading ratings in chunks...")
    ratings_chunks = []
    ratings_iter = pd.read_csv(
        ratings_path,
        chunksize=1_000_000,
        encoding="utf-8",
    )

    for i, chunk in enumerate(ratings_iter):
        ratings_chunks.append(chunk)
        print(f"  Chunk {i + 1} loaded: {len(chunk)} rows")

    ratings_df = pd.concat(ratings_chunks, ignore_index=True)
    print(f"✓ {len(ratings_df)} total ratings loaded")

    # Load tags (optional)
    try:
        print("Loading tags...")
        tags_path = os.path.join(data_dir, "tags.csv")
        if os.path.exists(tags_path):
            tags_chunks = []
            tags_iter = pd.read_csv(
                tags_path,
                chunksize=1_000_000,
                encoding="utf-8",
            )
            for i, chunk in enumerate(tags_iter):
                tags_chunks.append(chunk)
                print(f"  Tag chunk {i + 1} loaded: {len(chunk)} rows")
            tags_df = pd.concat(tags_chunks, ignore_index=True)
            print(f"✓ {len(tags_df)} total tags loaded")
        else:
            tags_df = pd.DataFrame()
            print("⚠️ tags.csv not found (optional)")
    except Exception as e:
        tags_df = pd.DataFrame()
        print(f"⚠️ No tags file or error: {e}")

    # Load links (optional)
    try:
        links_path = os.path.join(data_dir, "links.csv")
        if os.path.exists(links_path):
            links_df = pd.read_csv(links_path, encoding="utf-8")
            print(f"✓ {len(links_df)} links loaded")
        else:
            links_df = pd.DataFrame()
            print("⚠️ links.csv not found (optional)")
    except Exception as e:
        links_df = pd.DataFrame()
        print(f"⚠️ No links file or error: {e}")

    print("=" * 60 + "\n")
    return movies_df, ratings_df, tags_df, links_df


# ============================================================================
# INITIALIZE DATA
# ============================================================================

print("🎬 Initializing MOVIEFLIX...")
load_image_cache()
load_descriptions_cache()

try:
    MOVIES_DF, RATINGS_DF, TAGS_DF, LINKS_DF = load_movielens_data()
except FileNotFoundError as e:
    print(f"\n{e}")
    print("\n" + "=" * 60)
    print("⚠️  SETUP INSTRUCTIONS")
    print("=" * 60)
    print("1. Download MovieLens dataset from:")
    print("   https://files.grouplens.org/datasets/movielens/ml-32m.zip")
    print("   OR")
    print("   https://files.grouplens.org/datasets/movielens/ml-latest.zip")
    print("\n2. Extract the ZIP file to the current directory")
    print("   You should have a folder: ./ml-32m/ or ./ml-latest/")
    print("\n3. The folder should contain:")
    print("   - movies.csv (required)")
    print("   - ratings.csv (required)")
    print("   - links.csv (optional)")
    print("   - tags.csv (optional)")
    print("=" * 60 + "\n")
    raise

AVG_RATINGS = RATINGS_DF.groupby("movieId")["rating"].mean().to_dict()
RATING_COUNTS = RATINGS_DF.groupby("movieId")["rating"].count().to_dict()

ALL_GENRES = set()
for genres_str in MOVIES_DF["genres"]:
    if pd.notna(genres_str) and genres_str != "(no genres listed)":
        ALL_GENRES.update(genres_str.split("|"))
ALL_GENRES = sorted(list(ALL_GENRES))

movies_csv_path = (
    f"{DATA_DIR}/movies.csv" if os.path.exists(f"{DATA_DIR}/movies.csv") else None
)
RATINGS_MANAGER = UserRatingsManager(
    db_path="user_ratings.db",
    movies_csv_path=movies_csv_path,
)
CURRENT_USER = "default_user"

print("✅ Data loaded successfully!")


# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def search_movies(query):
    """Search movies by title."""
    if not query or len(query) < 2:
        return []

    query_lower = query.lower()
    filtered = MOVIES_DF[
        MOVIES_DF["title"].str.lower().str.contains(query_lower, na=False)
    ]

    results = []
    for _, row in filtered.head(50).iterrows():
        results.append((row["title"], row["movieId"]))

    return results


def get_movies_by_genre(genre=None, limit=18):
    """Get movies filtered by genre."""
    if genre and genre != "All Genres":
        filtered = MOVIES_DF[
            MOVIES_DF["genres"].str.contains(genre, case=False, na=False)
        ]
    else:
        filtered = MOVIES_DF

    movies = []
    for _, row in filtered.head(limit).iterrows():
        movie_id = row["movieId"]
        rating = AVG_RATINGS.get(movie_id, 0)
        movies.append(
            {
                "movieId": movie_id,
                "title": row["title"],
                "genres": row["genres"],
                "rating": rating,
                "image": get_tmdb_image(row["title"], movie_id),
            }
        )
    return movies


def get_movie_details(movie_id):
    """Get detailed information about a movie."""
    movie = MOVIES_DF[MOVIES_DF["movieId"] == movie_id].iloc[0]
    rating = AVG_RATINGS.get(movie_id, 0)
    num_ratings = RATING_COUNTS.get(movie_id, 0)

    image, description = get_tmdb_image_and_description(
        movie["title"], movie_id
    )

    return {
        "movieId": movie_id,
        "title": movie["title"],
        "genres": movie["genres"],
        "rating": rating,
        "num_ratings": num_ratings,
        "image": image,
        "description": description,
    }


def get_user_rating_for_movie(movie_id, username=CURRENT_USER):
    """Get user's rating for a specific movie."""
    ratings = RATINGS_MANAGER.get_user_ratings(username)
    for mid, rating, _, _ in ratings:
        if mid == movie_id:
            return rating
    return 0


def get_recommendations_for_user(username=CURRENT_USER, top_n=18):
    """Get recommendations based on user ratings."""
    user_ratings = RATINGS_MANAGER.get_user_ratings(username)

    if len(user_ratings) == 0:
        return []

    if item_factors is None or user_factors is None:
        return []

    rated_movie_ids = {movie_id for movie_id, _, _, _ in user_ratings}

    dummy_ratings = []
    for movie_id, rating, _, _ in user_ratings:
        if movie_id in movieid_to_idx:
            idx = movieid_to_idx[movie_id]
            dummy_ratings.append((idx, rating))

    if len(dummy_ratings) == 0:
        return []

    dummy_user_factors, _ = compute_dummy_user(
        item_factors,
        item_biases,
        dummy_ratings,
        K=15,
        lambda_val=0.1,
        tau=1.9,
        gamma_bias=0.04,
    )

    num_items = item_factors.shape[0]
    predictions = []

    for item_idx in range(num_items):
        movie_id = idx_to_movie[item_idx]

        if movie_id and movie_id in rated_movie_ids:
            continue

        score = np.dot(dummy_user_factors, item_factors[item_idx]) + 0.05 * item_biases[
            item_idx
        ]
        predictions.append((item_idx, score))

    predictions.sort(key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for item_idx, score in predictions[:top_n]:
        movie_id = idx_to_movie[item_idx]
        if movie_id and movie_id in MOVIES_DF["movieId"].values:
            movie = MOVIES_DF[MOVIES_DF["movieId"] == movie_id].iloc[0]
            rating = AVG_RATINGS.get(movie_id, 0)
            recommended_movies.append(
                {
                    "movieId": movie_id,
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "rating": rating,
                    "image": get_tmdb_image(movie["title"], movie_id),
                }
            )

    return recommended_movies


def is_movie_recommended(movie_id, username=CURRENT_USER, top_n=18):
    """Check if a movie is in the user's recommendations."""
    recs = get_recommendations_for_user(username, top_n=top_n)
    rec_ids = {m["movieId"] for m in recs}
    return movie_id in rec_ids


def get_why_recommended(movie_id, username=CURRENT_USER, top_n=3):
    """Get explanation for why a movie was recommended."""
    try:
        if movie_id not in movieid_to_idx:
            return None

        item_idx = movieid_to_idx[movie_id]

        user_ratings = RATINGS_MANAGER.get_user_ratings(username)
        dummy_ratings = []
        for mid, rating, _, _ in user_ratings:
            if mid in movieid_to_idx:
                idx = movieid_to_idx[mid]
                dummy_ratings.append((idx, rating))

        if len(dummy_ratings) == 0:
            return None

        contribs = top_historical_contributions(
            item_idx=item_idx,
            V=item_factors,
            user_history=dummy_ratings,
            lambda_val=0.1,
            tau=1.9,
        )

        top_movies = []
        for j, contrib in contribs[:top_n]:
            if j in idx_to_movie:
                contrib_movie_id = idx_to_movie[j]
                if contrib_movie_id in MOVIES_DF["movieId"].values:
                    movie = MOVIES_DF[MOVIES_DF["movieId"] == contrib_movie_id].iloc[0]
                    user_rating = get_user_rating_for_movie(
                        contrib_movie_id, username
                    )
                    top_movies.append(
                        {
                            "title": movie["title"],
                            "rating": user_rating,
                            "contribution": contrib,
                        }
                    )

        return top_movies if top_movies else None

    except Exception as e:
        print(f"Error in get_why_recommended: {e}")
        return None


def get_rated_movies(username=CURRENT_USER):
    """Get all movies rated by the user."""
    user_ratings = RATINGS_MANAGER.get_user_ratings(username)

    rated_movies = []
    for movie_id, rating, _, _ in user_ratings:
        if movie_id in MOVIES_DF["movieId"].values:
            movie = MOVIES_DF[MOVIES_DF["movieId"] == movie_id].iloc[0]
            avg_rating = AVG_RATINGS.get(movie_id, 0)
            rated_movies.append(
                {
                    "movieId": movie_id,
                    "title": movie["title"],
                    "genres": movie["genres"],
                    "rating": avg_rating,
                    "user_rating": rating,
                    "image": get_tmdb_image(movie["title"], movie_id),
                }
            )

    rated_movies.sort(key=lambda x: x["user_rating"], reverse=True)
    return rated_movies


# ============================================================================
# HTML GENERATION
# ============================================================================

def create_movie_grid_html(movies, page_type="popular", show_user_rating=False):
    """Returns HTML and list of movie cards components."""
    if page_type == "recommendations" and not movies:
        empty_state_html = """
        <div style="background: #0a0a0a; min-height: 60vh; color: #fff; font-family: 'Roboto', Arial, sans-serif; padding: 80px 60px; text-align: center;">
            <div style="max-width: 600px; margin: 0 auto;">
                <div style="font-size: 80px; margin-bottom: 20px;">🎬</div>
                <h2 style="font-size: 28px; font-weight: 700; margin-bottom: 15px; color: #e50914;">
                    Start Rating Movies to Get Recommendations
                </h2>
                <p style="font-size: 16px; color: #999; line-height: 1.6;">
                    Browse popular movies below and rate them to receive personalized recommendations based on your taste!
                </p>
            </div>
        </div>
        """  # noqa: E501
        return empty_state_html, []

    if not movies:
        return (
            "<h2 style='text-align: center; color: #999; padding: 100px;'>"
            "No movies found</h2>",
            [],
        )

    page_titles = {
        "popular": "🔥 Popular Movies",
        "recommendations": "🎯 Your Personalized Recommendations",
        "rated": "⭐ Your Rated Movies",
    }

    html = f"""
    <div style="background: #0a0a0a; color: #fff; font-family: 'Roboto', Arial, sans-serif; padding: 40px 60px;">
        <h2 style="font-size: 32px; font-weight: 700; margin-bottom: 30px; color: #e50914;">
            {page_titles.get(page_type, "Movies")}
        </h2>
    </div>
    """

    return html, movies


def create_movie_card_html(movie, page_type="popular", show_user_rating=False):
    """Create individual movie card with rating badge."""
    rating_display = ""
    if page_type in ["popular", "recommendations"]:
        num_ratings = RATING_COUNTS.get(movie["movieId"], 0)
        rating_display = f'<div style="color: #999; font-size: 12px; margin-top: 5px;">⭐ {num_ratings} ratings</div>'

    user_rating_badge = ""
    if show_user_rating and "user_rating" in movie:
        user_stars = "⭐" * int(movie["user_rating"]) + "☆" * (
            5 - int(movie["user_rating"])
        )
        user_rating_badge = f'''
        <div style="
            position: absolute; top: 10px; right: 10px;
            background: rgba(229,9,20,0.95); color: white;
            padding: 6px 10px; border-radius: 12px;
            font-size: 12px; font-weight: 700;
            box-shadow: 0 2px 8px rgba(0,0,0,0.5);
            z-index: 10;
        ">
            {user_stars} {movie['user_rating']}/5
        </div>
        '''
    elif page_type == "recommendations":
        user_rating = get_user_rating_for_movie(movie["movieId"])
        if user_rating > 0:
            user_stars = "⭐" * int(user_rating) + "☆" * (5 - int(user_rating))
            user_rating_badge = f'''
            <div style="
                position: absolute; top: 10px; right: 10px;
                background: rgba(229,9,20,0.95); color: white;
                padding: 6px 10px; border-radius: 12px;
                font-size: 12px; font-weight: 700;
                box-shadow: 0 2px 8px rgba(0,0,0,0.5);
                z-index: 10;
            ">
                {user_stars} {user_rating}/5
            </div>
            '''

    html = f"""
    <div style="
        border-radius: 12px; overflow: hidden;
        background: #1a1a1a; transition: transform 0.3s;
        position: relative; height: 100%;
    ">
        {user_rating_badge}
        <img src="{movie['image']}" style="width: 100%; height: 300px; object-fit: cover;"
             onerror="this.src='https://via.placeholder.com/300x450/1a1a2e/fff?text=No+Image'">
        <div style="padding: 15px;">
            <div style="font-size: 15px; font-weight: 700; margin-bottom: 8px; height: 40px; overflow: hidden; color: #ffffff;">
                {movie['title']}
            </div>
            {rating_display}
        </div>
    </div>
    """  # noqa: E501

    return html


def create_detail_html(movie_id):
    """Create detailed movie view HTML."""
    movie = get_movie_details(movie_id)
    genres_list = movie["genres"].split("|") if movie["genres"] else []

    user_rating = get_user_rating_for_movie(movie_id)
    user_rating_display = ""
    if user_rating > 0:
        user_stars = "⭐" * int(user_rating) + "☆" * (5 - int(user_rating))
        user_rating_display = f"""
        <div style="
            margin-top: 15px; padding: 12px 18px;
            background: rgba(229,9,20,0.15); border: 2px solid #e50914;
            border-radius: 12px; display: inline-block;
        ">
            <span style="color: #ffd700; font-weight: 700; font-size: 16px;">
                Your Rating: {user_stars} {user_rating}/5
            </span>
        </div>
        """

    description_section = ""
    if movie.get("description"):
        description_section = f"""
        <div style="max-width:1000px; margin:30px auto; padding:30px; background:#111; border-radius:18px;">
            <h2 style="font-size:26px; font-weight:700; margin-bottom:18px; color:#e50914;">
                📖 Synopsis
            </h2>
            <p style="font-size:16px; line-height:1.8; color:#d6d6d6;">
                {movie['description']}
            </p>
        </div>
        """

    is_recommended = is_movie_recommended(movie_id)

    why_for_you_section = ""
    if is_recommended:
        why_movies = get_why_recommended(movie_id)
        if why_movies:
            movies_list = []
            for m in why_movies:
                stars = "⭐" * int(m["rating"]) + "☆" * (5 - int(m["rating"]))
                movies_list.append(
                    f"<strong>{m['title']}</strong> ({stars} {m['rating']}/5)"
                )

            if movies_list:
                movies_text = (
                    ", ".join(movies_list[:-1])
                    + (" and " if len(movies_list) > 1 else "")
                    + movies_list[-1]
                )
            else:
                movies_text = ""

            why_for_you_section = f"""
            <div style="max-width:1000px; margin:30px auto; padding:30px; background:linear-gradient(135deg, #1a1a1a 0%, #2a0a0a 100%); border-radius:18px; border: 2px solid rgba(229,9,20,0.3);">
                <h2 style="font-size:26px; font-weight:700; margin-bottom:18px; color:#e50914;">
                    💡 Why For You
                </h2>
                <p style="font-size:16px; line-height:1.8; color:#ffffff;">
                    You liked {movies_text}, so you'll surely like this movie!
                </p>
            </div>
            """  # noqa: E501

    recommendation_section = ""
    if is_recommended:
        recommendation_section = f"""
        <div style="max-width:1000px; margin:30px auto; padding:40px; background:#111; border-radius:18px;">
            <h2 style="font-size:26px; font-weight:700; margin-bottom:22px; color:#e50914;">
                💡 Recommendation Info
            </h2>

            <div style="font-size:16px; line-height:1.9; color:#ffffff; background:#0d0d0d; padding:28px; border-radius:14px;">
                <ul style="padding-left:22px; line-height:2.1;">
                    <li><strong>Genres:</strong> {movie['genres'].replace('|', ', ')}</li>
                    <li><strong>Total Ratings:</strong> {movie['num_ratings']} ratings</li>
                    <li><strong>System:</strong> ALS (Alternating Least Squares) Recommendation</li>
                </ul>
            </div>
        </div>
        """  # noqa: E501

    genres_html = "".join(
        f'<span style="padding:6px 14px; background:#151515; border:1px solid rgba(229,9,20,0.6); border-radius:18px; font-size:13px; color:#e5e5e5;">{g}</span>'  # noqa: E501
        for g in genres_list
    )

    return f"""
    <div style="background:#0b0b0b; min-height:100vh; color:#f2f2f2; font-family:'Inter','Roboto',Arial,sans-serif;">
        <div style="position:relative; height:42vh; overflow:hidden;">
            <div style="
                position:absolute; inset:0;
                background: linear-gradient(to right, rgba(0,0,0,0.85) 45%, rgba(0,0,0,0.25)),
                    url('{movie['image']}') center/cover;
                filter: blur(14px); transform: scale(1.12); z-index:0;
            "></div>

            <div style="position:relative; z-index:1; display:flex; align-items:center; height:100%; padding:60px;">
                <img src="{movie['image']}" style="
                    width:190px; border-radius:14px;
                    box-shadow:0 25px 60px rgba(0,0,0,0.9);
                ">

                <div style="margin-left:48px; max-width:720px;">
                    <h1 style="font-size:44px; font-weight:800; margin-bottom:16px; color:#ffffff;">
                        {movie['title']}
                    </h1>

                    <div style="display:flex; gap:24px; margin-bottom:18px; font-size:17px; color:#d0d0d0;">
                        <span>⭐ {movie['num_ratings']} ratings</span>
                    </div>

                    <div style="display:flex; gap:10px; flex-wrap:wrap;">
                        {genres_html}
                    </div>
                    {user_rating_display}
                </div>
            </div>
        </div>

        {description_section}

        {why_for_you_section}

        {recommendation_section}
    </div>
    """


def update_stats_html():
    """Generate user statistics HTML."""
    user_stats = RATINGS_MANAGER.get_user_stats(CURRENT_USER)
    return f"""
    <div style="
        background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
        border-radius: 12px; padding: 20px; color: #e0e0e0;
        border: 1px solid rgba(229,9,20,0.3);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4); margin-bottom: 20px;
    ">
        <h3 style="font-size: 18px; font-weight: 700; color: #ffd700; margin-bottom: 15px;">
            👤 Your Profile
        </h3>
        <div style="font-size: 14px; line-height: 2;">
            <div style="display: flex; justify-content: space-between; padding: 6px 0;">
                <span style="color: #bbb;">📊 Movies Rated:</span>
                <span style="color: #fff; font-weight: 600;">{user_stats['num_ratings']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-top: 1px solid #333; margin-top: 8px; padding-top: 8px;">
                <span style="color: #bbb;">⭐ Your Average:</span>
                <span style="color: #ffd700; font-weight: 700;">{user_stats['avg_rating']}/5</span>
            </div>
        </div>
    </div>
    """  # noqa: E501


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

custom_css = """
button { font-family: 'Roboto', Arial, sans-serif !important; }

.sidebar-button {
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
    border: 1.5px solid #333 !important;
    color: #fff !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-bottom: 10px !important;
}

.sidebar-button:hover {
    background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%) !important;
    border-color: #e50914 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(229, 9, 20, 0.5) !important;
}

.genre-button {
    background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%) !important;
    border: 1px solid #444 !important;
    color: #bbb !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border-radius: 20px !important;
    transition: all 0.3s ease !important;
    min-width: 100px !important;
}

.genre-button:hover {
    background: linear-gradient(135deg, #e50914 0%, #b00710 100%) !important;
    border-color: #e50914 !important;
    color: #fff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4) !important;
}

.genre-button.selected {
    background: linear-gradient(135deg, #e50914 0%, #b00710 100%) !important;
    border-color: #e50914 !important;
    color: #fff !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.4) !important;
}

input[type="range"] {
    -webkit-appearance: none !important;
    appearance: none !important;
    background: linear-gradient(to right, #e50914 0%, #e50914 60%, #333 60%, #333 100%) !important;
    height: 8px !important;
    border-radius: 5px !important;
}

input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    background: #e50914 !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(229, 9, 20, 0.5) !important;
}

input[type="range"]::-moz-range-thumb {
    width: 20px !important;
    height: 20px !important;
    border-radius: 50% !important;
    background: #e50914 !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(229, 9, 20, 0.5) !important;
    border: none !important;
}

button[size="sm"] {
    background: linear-gradient(135deg, #e50914 0%, #b00710 100%) !important;
    border: 1px solid #e50914 !important;
    color: #fff !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    margin-top: 8px !important;
}

button[size="sm"]:hover {
    background: linear-gradient(135deg, #ff0a1f 0%, #c90913 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.6) !important;
}
"""

with gr.Blocks(
    title="🎬 MOVIEFLIX", theme=gr.themes.Soft(), css=custom_css
) as app_scale:
    gr.Markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a1a 0%, #222222 100%);
        padding: 18px 24px; text-align: center; border-radius: 14px;
        margin-bottom: 18px; border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    ">
        <h1 style="color: #e50914; font-size: 36px; margin: 0; font-weight: 900;">
            🎬 MOVIEFLIX
        </h1>
        <p style="color: #cccccc; font-size: 15px; margin-top: 6px;">
            Personalized Movie Recommendation System<br>
            <span style="color:#aaaaaa; font-style: italic;">MovieLens + TMDB • ALS Recommender</span>
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
                border-radius: 12px; padding: 20px; margin-bottom: 20px;
                border: 1px solid rgba(229,9,20,0.3);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            ">
                <h2 style="font-size: 20px; font-weight: 800; color: #e50914; margin-bottom: 20px;">
                    🧭 NAVIGATION
                </h2>
            </div>
            """)

            popular_button = gr.Button(
                "🔥 Popular Movies", elem_classes="sidebar-button"
            )
            recommendations_button = gr.Button(
                "🎯 My Recommendations", elem_classes="sidebar-button"
            )
            rated_movies_button = gr.Button(
                "⭐ My Rated Movies", elem_classes="sidebar-button"
            )

            stats_html = gr.HTML(update_stats_html())

            gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
                border-radius: 12px; padding: 20px; color: #e0e0e0; margin-bottom: 20px;
                border: 1px solid rgba(229,9,20,0.3);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            ">
                <h2 style="font-size: 20px; font-weight: 800; color: #e50914; margin-bottom: 15px;">
                    🔍 SEARCH MOVIE
                </h2>
            </div>
            """)

            movie_search = gr.Textbox(
                label="", placeholder="Search movie by title...", scale=1
            )

            movie_dropdown = gr.Dropdown(
                choices=[], label="", visible=False
            )

            view_details_button = gr.Button(
                "👁️ View Details", variant="primary", visible=False
            )

            gr.Markdown("<br>")

            gr.HTML("""
            <div style="
                background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%);
                border-radius: 12px; padding: 20px; margin-top: 15px;
                border: 1px solid rgba(229,9,20,0.3);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            ">
                <h3 style="font-size: 18px; font-weight: 700; color: #ffd700; margin-bottom: 15px;">
                    ⭐ Rate this Movie
                </h3>
                <p style="font-size: 13px; color: #aaa; line-height: 1.6; margin-top: 10px;">
                    💡 <strong>How to use:</strong><br>
                    • Click on a movie poster to view details<br>
                    • Add or update your rating using the slider<br>
                    • Click "Delete Rating" button below to remove ratings
                </p>
            </div>
            """)

            rating_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=0.5,
                label="Your Rating (1-5 ⭐)",
                visible=False,
                value=3,
            )

            submit_rating_button = gr.Button(
                "💾 Save Rating", variant="primary", visible=False
            )

            delete_rating_button = gr.Button(
                "🗑️ Delete Rating", variant="secondary", visible=False
            )

        with gr.Column(scale=4):
            with gr.Row():
                gr.Markdown("""
                <div style="
                    background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
                    padding: 15px 25px; border-radius: 12px; margin-bottom: 20px;
                    border: 1px solid rgba(229,9,20,0.2);
                    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
                ">
                    <span style="color: #e50914; font-weight: 700; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">
                        🎭 FILTER BY GENRE
                    </span>
                </div>
                """)  # noqa: E501

            with gr.Row():
                genre_buttons = []
                all_genres_btn = gr.Button(
                    "All Genres", elem_classes="genre-button selected"
                )
                genre_buttons.append(("All Genres", all_genres_btn))

                for genre in ALL_GENRES[:8]:
                    btn = gr.Button(genre, elem_classes="genre-button")
                    genre_buttons.append((genre, btn))

            initial_movies = get_movies_by_genre(limit=18)

            initial_header, _ = create_movie_grid_html(
                initial_movies, "popular"
            )
            content_header = gr.HTML(value=initial_header)

            movie_grid_rows = []
            movie_index = 0
            for i in range(3):
                with gr.Row():
                    row_columns = []
                    for j in range(6):
                        with gr.Column(scale=1, min_width=180):
                            if movie_index < len(initial_movies):
                                movie = initial_movies[movie_index]
                                card_html = create_movie_card_html(
                                    movie, "popular", show_user_rating=False
                                )
                                movie_html = gr.HTML(value=card_html)
                                movie_btn = gr.Button(
                                    "👁️ View Details",
                                    size="sm",
                                    variant="secondary",
                                    visible=True,
                                )
                            else:
                                movie_html = gr.HTML(value="")
                                movie_btn = gr.Button(
                                    "👁️ View Details",
                                    size="sm",
                                    variant="secondary",
                                    visible=False,
                                )
                            row_columns.append((movie_html, movie_btn))
                            movie_index += 1
                    movie_grid_rows.append(row_columns)

    current_view = gr.State("popular")
    current_page = gr.State("popular")
    current_movie_id = gr.State(None)
    current_genre = gr.State("All Genres")
    selected_movie_from_search = gr.State(None)

    movie_id_states = []
    for i in range(18):
        if i < len(initial_movies):
            movie_id_state = gr.State(value=initial_movies[i]["movieId"])
        else:
            movie_id_state = gr.State(value=None)
        movie_id_states.append(movie_id_state)

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    def update_search_results(query):
        """Update search dropdown based on query."""
        if not query or len(query) < 2:
            return (
                gr.update(choices=[], visible=False),
                gr.update(visible=False),
                None,
            )

        results = search_movies(query)
        if results:
            return (
                gr.update(choices=results, visible=True),
                gr.update(visible=True),
                None,
            )
        return (
            gr.update(choices=[], visible=False),
            gr.update(visible=False),
            None,
        )

    def select_movie_from_dropdown(movie_id):
        """Select movie from dropdown."""
        return movie_id

    def view_selected_movie(selected_movie, current_page_state, genre):
        """View movie details from search."""
        if selected_movie:
            return show_movie_detail(selected_movie, current_page_state, genre)
        return show_popular(genre)

    def show_popular(genre="All Genres"):
        """Show popular movies page."""
        movies = get_movies_by_genre(
            genre=genre if genre != "All Genres" else None, limit=18
        )
        header_html, _ = create_movie_grid_html(movies, "popular")

        updates = [update_stats_html(), header_html]
        for i in range(18):
            if i < len(movies):
                movie = movies[i]
                card_html = create_movie_card_html(
                    movie, "popular", show_user_rating=False
                )
                updates.append(card_html)
                updates.append(gr.update(visible=True))
                updates.append(movie["movieId"])
            else:
                updates.append("")
                updates.append(gr.update(visible=False))
                updates.append(None)

        updates.extend([
            "popular",
            "home",
            None,
            genre,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ])
        return updates

    def show_recommendations():
        """Show recommendations page."""
        movies = get_recommendations_for_user(CURRENT_USER, top_n=18)
        header_html, _ = create_movie_grid_html(movies, "recommendations")

        updates = [update_stats_html(), header_html]
        for i in range(18):
            if i < len(movies):
                movie = movies[i]
                card_html = create_movie_card_html(
                    movie, "recommendations", show_user_rating=False
                )
                updates.append(card_html)
                updates.append(gr.update(visible=True))
                updates.append(movie["movieId"])
            else:
                updates.append("")
                updates.append(gr.update(visible=False))
                updates.append(None)

        updates.extend([
            "recommendations",
            "home",
            None,
            "All Genres",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ])
        return updates

    def show_rated_movies():
        """Show rated movies page."""
        movies = get_rated_movies(CURRENT_USER)
        header_html, _ = create_movie_grid_html(
            movies, "rated", show_user_rating=True
        )

        updates = [update_stats_html(), header_html]
        for i in range(18):
            if i < len(movies):
                movie = movies[i]
                card_html = create_movie_card_html(
                    movie, "rated", show_user_rating=True
                )
                updates.append(card_html)
                updates.append(gr.update(visible=True))
                updates.append(movie["movieId"])
            else:
                updates.append("")
                updates.append(gr.update(visible=False))
                updates.append(None)

        updates.extend([
            "rated",
            "home",
            None,
            "All Genres",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        ])
        return updates

    def show_movie_detail(movie_id, current_page_state, genre):
        """Show movie detail page."""
        if movie_id is None:
            return show_popular(genre)

        user_rating = get_user_rating_for_movie(movie_id)
        detail_html = create_detail_html(movie_id)

        updates = [update_stats_html(), detail_html]
        for i in range(18):
            updates.append("")
            updates.append(gr.update(visible=False))
            updates.append(None)

        updates.extend([
            current_page_state,
            "detail",
            movie_id,
            genre,
            gr.update(
                visible=True, value=user_rating if user_rating > 0 else 3
            ),
            gr.update(visible=True),
            gr.update(visible=True if user_rating > 0 else False),
        ])
        return updates

    def handle_rating_submission(
        movie_id, rating_value, current_page_state, genre
    ):
        """Handle rating submission."""
        if movie_id and rating_value:
            movie = get_movie_details(movie_id)
            RATINGS_MANAGER.add_rating(movie_id, rating_value, CURRENT_USER)
            gr.Info(
                f"✅ Rating saved: {rating_value}/5 for {movie['title']}"
            )

            return show_movie_detail(movie_id, current_page_state, genre)

        return show_popular(genre)

    def handle_delete_rating(movie_id, current_page_state, genre):
        """Handle rating deletion."""
        if movie_id:
            movie = get_movie_details(movie_id)
            RATINGS_MANAGER.delete_rating(movie_id, CURRENT_USER)
            gr.Info(f"🗑️ Rating deleted for {movie['title']}")

            if current_page_state == "rated":
                return show_rated_movies()
            return show_movie_detail(movie_id, current_page_state, genre)
        return show_popular(genre)

    all_movie_outputs = [stats_html, content_header]
    movie_buttons = []
    for i, row in enumerate(movie_grid_rows):
        for j, (movie_html, movie_btn) in enumerate(row):
            all_movie_outputs.append(movie_html)
            all_movie_outputs.append(movie_btn)
            all_movie_outputs.append(
                movie_id_states[i * 6 + j]
            )
            movie_buttons.append((movie_btn, movie_id_states[i * 6 + j]))

    all_movie_outputs.extend([
        current_page,
        current_view,
        current_movie_id,
        current_genre,
        rating_slider,
        submit_rating_button,
        delete_rating_button,
    ])

    popular_button.click(
        fn=lambda g: show_popular(g),
        inputs=[current_genre],
        outputs=all_movie_outputs,
    )

    recommendations_button.click(
        fn=show_recommendations, outputs=all_movie_outputs
    )

    rated_movies_button.click(
        fn=show_rated_movies, outputs=all_movie_outputs
    )

    for genre_name, genre_btn in genre_buttons:

        def make_genre_click_handler(g):
            return lambda: show_popular(g)

        genre_btn.click(
            fn=make_genre_click_handler(genre_name),
            outputs=all_movie_outputs,
        )

    for i, (movie_btn, movie_id_state) in enumerate(movie_buttons):
        movie_btn.click(
            fn=show_movie_detail,
            inputs=[movie_id_state, current_page, current_genre],
            outputs=all_movie_outputs,
        )

    movie_search.change(
        fn=update_search_results,
        inputs=[movie_search],
        outputs=[
            movie_dropdown,
            view_details_button,
            selected_movie_from_search,
        ],
    )

    movie_dropdown.change(
        fn=select_movie_from_dropdown,
        inputs=[movie_dropdown],
        outputs=[selected_movie_from_search],
    )

    view_details_button.click(
        fn=view_selected_movie,
        inputs=[selected_movie_from_search, current_page, current_genre],
        outputs=all_movie_outputs,
    )

    submit_rating_button.click(
        fn=handle_rating_submission,
        inputs=[current_movie_id, rating_slider, current_page, current_genre],
        outputs=all_movie_outputs,
    )

    delete_rating_button.click(
        fn=handle_delete_rating,
        inputs=[current_movie_id, current_page, current_genre],
        outputs=all_movie_outputs,
    )

    gr.Markdown("""
    <div style='text-align: center; padding: 20px; color: #666; margin-top: 40px;'>
        <p>💡 <strong>How it works:</strong> Browse movies • Rate them • Get personalized recommendations</p>
        <p>📊 Data: MovieLens | 🖼️ Images: TMDB API | 💾 Ratings saved locally in SQLite</p>
    </div>
    """)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app_scale.launch(server_name="0.0.0.0", server_port=port)
