import csv
import os
import sqlite3


class UserRatingsManager:
    """Gestionnaire de base de données pour les ratings utilisateurs"""

    def __init__(self, db_path="user_ratings.db", movies_csv_path="ml-32m/movies.csv"):
        self.db_path = db_path
        self.movies_csv_path = movies_csv_path
        self.conn = None
        self.init_database()
        self.load_movies_mapping()

    def init_database(self):
        """Initialise la base de données SQLite"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.conn.cursor()

        # Table des utilisateurs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Table des ratings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_ratings (
                rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, movie_id)
            )
        ''')

        # Table des films aimés (likes)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_likes (
                like_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, movie_id)
            )
        ''')

        # Table de la liste personnelle
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_watchlist (
                watchlist_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(user_id, movie_id)
            )
        ''')

        # Index pour améliorer les performances
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_ratings ON user_ratings(user_id, movie_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_likes ON user_likes(user_id, movie_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_watchlist ON user_watchlist(user_id, movie_id)')

        self.conn.commit()
        print("✓ Base de données initialisée")

    def load_movies_mapping(self):
        """Charge le mapping movieId -> titre depuis movies.csv"""
        self.movieid_to_title = {}
        self.title_to_movieid = {}

        if not os.path.exists(self.movies_csv_path):
            print(f"⚠️ Fichier {self.movies_csv_path} non trouvé")
            return

        with open(self.movies_csv_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                movie_id = int(row[0])
                title = row[1]
                self.movieid_to_title[movie_id] = title
                self.title_to_movieid[title] = movie_id

        print(f"✓ {len(self.movieid_to_title)} films chargés dans le mapping")

    # ============================================================================
    # GESTION DES UTILISATEURS
    # ============================================================================

    def create_user(self, username="default_user"):
        """Crée un nouvel utilisateur"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username) VALUES (?)', (username,))
            self.conn.commit()
            user_id = cursor.lastrowid
            print(f"✓ Utilisateur '{username}' créé avec ID: {user_id}")
            return user_id
        except sqlite3.IntegrityError:
            # L'utilisateur existe déjà
            cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
            user_id = cursor.fetchone()[0]
            print(f"✓ Utilisateur '{username}' déjà existant avec ID: {user_id}")
            return user_id

    def get_user_id(self, username="default_user"):
        """Récupère l'ID d'un utilisateur ou le crée s'il n'existe pas"""
        cursor = self.conn.cursor()  # type: ignore
        cursor.execute('SELECT user_id FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()

        if result:
            return result[0]
        else:
            return self.create_user(username)

    # ============================================================================
    # GESTION DES RATINGS
    # ============================================================================

    def add_rating(self, movie_id, rating, username="default_user"):
        """Ajoute ou met à jour un rating"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO user_ratings (user_id, movie_id, rating)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, movie_id)
                DO UPDATE SET rating = ?, updated_at = CURRENT_TIMESTAMP
            ''', (user_id, movie_id, rating, rating))
            self.conn.commit()  # type: ignore

            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            print(f"✓ Rating {rating}/5 ajouté pour '{movie_title}'")
            return True
        except Exception as e:
            print(f"❌ Erreur lors de l'ajout du rating: {e}")
            return False

    def get_user_ratings(self, username="default_user"):
        """Récupère tous les ratings d'un utilisateur"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT movie_id, rating, created_at, updated_at
            FROM user_ratings
            WHERE user_id = ?
            ORDER BY updated_at DESC
        ''', (user_id,))

        ratings = cursor.fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in ratings]

    def get_user_ratings_for_recommendation(self, username="default_user"):
        """Récupère les ratings sous forme de liste pour le modèle de recommandation"""
        ratings = self.get_user_ratings(username)
        # Format: [(movie_id, rating), ...]
        return [(movie_id, rating) for movie_id, rating, _, _ in ratings]

    def delete_rating(self, movie_id, username="default_user"):
        """Supprime un rating"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            DELETE FROM user_ratings
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        self.conn.commit()

        movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
        print(f"✓ Rating supprimé pour '{movie_title}'")
        return True

    # ============================================================================
    # GESTION DES LIKES
    # ============================================================================

    def add_like(self, movie_id, username="default_user"):
        """Ajoute un like à un film"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO user_likes (user_id, movie_id)
                VALUES (?, ?)
            ''', (user_id, movie_id))
            self.conn.commit()

            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            print(f"✓ Like ajouté pour '{movie_title}'")
            return True
        except sqlite3.IntegrityError:
            print("⚠️ Film déjà liké")
            return False

    def remove_like(self, movie_id, username="default_user"):
        """Retire un like"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            DELETE FROM user_likes
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        self.conn.commit()

        movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
        print(f"✓ Like retiré pour '{movie_title}'")
        return True

    def get_user_likes(self, username="default_user"):
        """Récupère tous les films likés par un utilisateur"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT movie_id, created_at
            FROM user_likes
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))

        likes = cursor.fetchall()
        return [like[0] for like in likes]

    def is_liked(self, movie_id, username="default_user"):
        """Vérifie si un film est liké"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM user_likes
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))

        return cursor.fetchone()[0] > 0

    # ============================================================================
    # GESTION DE LA WATCHLIST
    # ============================================================================

    def add_to_watchlist(self, movie_id, username="default_user"):
        """Ajoute un film à la liste"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO user_watchlist (user_id, movie_id)
                VALUES (?, ?)
            ''', (user_id, movie_id))
            self.conn.commit()

            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            print(f"✓ Film ajouté à la liste: '{movie_title}'")
            return True
        except sqlite3.IntegrityError:
            print("⚠️ Film déjà dans la liste")
            return False

    def remove_from_watchlist(self, movie_id, username="default_user"):
        """Retire un film de la liste"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            DELETE FROM user_watchlist
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))
        self.conn.commit()

        movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
        print(f"✓ Film retiré de la liste: '{movie_title}'")
        return True

    def get_user_watchlist(self, username="default_user"):
        """Récupère la liste des films d'un utilisateur"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT movie_id, added_at
            FROM user_watchlist
            WHERE user_id = ?
            ORDER BY added_at DESC
        ''', (user_id,))

        watchlist = cursor.fetchall()
        return [item[0] for item in watchlist]

    def is_in_watchlist(self, movie_id, username="default_user"):
        """Vérifie si un film est dans la liste"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT COUNT(*) FROM user_watchlist
            WHERE user_id = ? AND movie_id = ?
        ''', (user_id, movie_id))

        return cursor.fetchone()[0] > 0

    # ============================================================================
    # STATISTIQUES ET UTILITAIRES
    # ============================================================================

    def get_user_stats(self, username="default_user"):
        """Récupère les statistiques d'un utilisateur"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        # Nombre de ratings
        cursor.execute('SELECT COUNT(*) FROM user_ratings WHERE user_id = ?', (user_id,))
        num_ratings = cursor.fetchone()[0]

        # Nombre de likes
        cursor.execute('SELECT COUNT(*) FROM user_likes WHERE user_id = ?', (user_id,))
        num_likes = cursor.fetchone()[0]

        # Nombre de films dans la watchlist
        cursor.execute('SELECT COUNT(*) FROM user_watchlist WHERE user_id = ?', (user_id,))
        num_watchlist = cursor.fetchone()[0]

        # Rating moyen
        cursor.execute('SELECT AVG(rating) FROM user_ratings WHERE user_id = ?', (user_id,))
        avg_rating = cursor.fetchone()[0] or 0

        return {
            'num_ratings': num_ratings,
            'num_likes': num_likes,
            'num_watchlist': num_watchlist,
            'avg_rating': round(avg_rating, 2)
        }

    def get_recent_activity(self, username="default_user", limit=10):
        """Récupère l'activité récente d'un utilisateur"""
        user_id = self.get_user_id(username)
        cursor = self.conn.cursor()

        # Combiner les activités récentes
        cursor.execute('''
            SELECT 'rating' as type, movie_id, rating as value, updated_at as timestamp
            FROM user_ratings
            WHERE user_id = ?
            UNION ALL
            SELECT 'like' as type, movie_id, NULL as value, created_at as timestamp
            FROM user_likes
            WHERE user_id = ?
            UNION ALL
            SELECT 'watchlist' as type, movie_id, NULL as value, added_at as timestamp
            FROM user_watchlist
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, user_id, user_id, limit))

        activities = cursor.fetchall()

        result = []
        for activity in activities:
            activity_type, movie_id, value, timestamp = activity
            movie_title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
            result.append({
                'type': activity_type,
                'movie_id': movie_id,
                'movie_title': movie_title,
                'value': value,
                'timestamp': timestamp
            })

        return result

    def clear_database(self):
        """Supprime toutes les données et les tables de la base"""
        cursor = self.conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS user_ratings")
        cursor.execute("DROP TABLE IF EXISTS user_likes")
        cursor.execute("DROP TABLE IF EXISTS user_watchlist")
        cursor.execute("DROP TABLE IF EXISTS users")
        self.conn.commit()
        print("✓ Base de données vidée")
        # Réinitialiser la DB pour pouvoir réutiliser les tables
        self.init_database()

    def export_user_ratings_csv(self, username="default_user", output_path="my_ratings.csv"):
        """Exporte les ratings d'un utilisateur en CSV"""
        ratings = self.get_user_ratings(username)

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['movieId', 'title', 'rating', 'created_at', 'updated_at'])

            for movie_id, rating, created_at, updated_at in ratings:
                title = self.movieid_to_title.get(movie_id, f"Movie {movie_id}")
                writer.writerow([movie_id, title, rating, created_at, updated_at])

        print(f"✓ Ratings exportés vers {output_path}")
        return output_path

    def close(self):
        """Ferme la connexion à la base de données"""
        if self.conn:
            self.conn.close()
            print("✓ Connexion à la base de données fermée")


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    # Initialiser le gestionnaire
    manager = UserRatingsManager()
    # Créer un utilisateur
    user_id = manager.create_user("john_doe")
    # Ajouter des ratings
    manager.add_rating(movie_id=1, rating=5.0, username="john_doe")
    manager.add_rating(movie_id=2, rating=4.5, username="john_doe")
    manager.add_rating(movie_id=3, rating=3.5, username="john_doe")
    # Ajouter des likes
    manager.add_like(movie_id=1, username="john_doe")
    manager.add_like(movie_id=5, username="john_doe")
    # Ajouter à la watchlist
    manager.add_to_watchlist(movie_id=10, username="john_doe")
    manager.add_to_watchlist(movie_id=15, username="john_doe")
    # Récupérer les ratings
    ratings = manager.get_user_ratings("john_doe")
    print(f"\n📊 Ratings de john_doe: {len(ratings)}")
    for movie_id, rating, created, updated in ratings:
        print(f"  - Movie {movie_id}: {rating}/5")

    # Récupérer les stats
    stats = manager.get_user_stats("john_doe")
    print("\n📈 Statistiques:")
    print(f"  - Ratings: {stats['num_ratings']}")
    print(f"  - Likes: {stats['num_likes']}")
    print(f"  - Watchlist: {stats['num_watchlist']}")
    print(f"  - Rating moyen: {stats['avg_rating']}/5")
    # Activité récente
    activities = manager.get_recent_activity("john_doe", limit=5)
    print("\n🕒 Activité récente:")
    for activity in activities:
        print(f"  - {activity['type']}: {activity['movie_title']} ({activity['timestamp']})")
    # Exporter les ratings
    manager.export_user_ratings_csv("john_doe")
    manager.clear_database()
    # Fermer la connexion
    manager.close()
