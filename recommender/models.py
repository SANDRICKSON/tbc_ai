# music_recommender/models.py
from django.db import models
from django.contrib.auth.models import User
import numpy as np
import joblib
import pandas as pd
import os
from django.conf import settings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

class Song(models.Model):
    MOOD_CHOICES = [
        ('happy', 'ბედნიერი'),
        ('sad', 'სევდიანი'),
        ('energetic', 'ენერგიული'),
        ('romantic', 'რომანტიკული'),
        ('chill', 'მშვიდი'),
        ('angry', 'გაბრაზებული'),
        ('focused', 'კონცენტრირებული'),
        ('party', 'ქეიფი'),
        ('workout', 'ვარჯიში'),
        ('travel', 'მგზავრობა'),
    ]

    GENRE_CHOICES = [
        ('pop', 'პოპი'),
        ('rock', 'როკი'),
        ('jazz', 'ჯაზი'),
        ('hiphop', 'ჰიპჰოპ'),
        ('electronic', 'ელექტრონული'),
        ('folk', 'ფოლკი'),
        ('classical', 'კლასიკური'),
        ('rb', 'R&B'),
        ('country', 'კანტრი'),
        ('metal', 'მეტალი'),
    ]

    title = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    genre = models.CharField(max_length=50, choices=GENRE_CHOICES)
    mood = models.CharField(max_length=50, choices=MOOD_CHOICES)
    energy = models.FloatField(default=0.5)
    valence = models.FloatField(default=0.5)
    danceability = models.FloatField(default=0.5)
    tempo = models.IntegerField(default=120)
    duration = models.IntegerField(default=180)
    youtube_link = models.URLField(blank=True, null=True)
    spotify_link = models.URLField(blank=True, null=True)
    audio_file = models.FileField(upload_to='songs/', blank=True, null=True)
    cover_image = models.ImageField(upload_to='covers/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} - {self.artist}"

    def get_features_array(self):
        return np.array([
            self.energy,
            self.valence,
            self.danceability,
            self.tempo / 200,
            self.duration / 600
        ])

class UserRating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    song = models.ForeignKey(Song, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=[(i, i) for i in range(1, 6)])
    mood_when_listened = models.CharField(max_length=50, choices=Song.MOOD_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['user', 'song']

    def __str__(self):
        return f"{self.user.username} - {self.song.title}: {self.rating}"

class PlayHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    song = models.ForeignKey(Song, on_delete=models.CASCADE)
    listened_at = models.DateTimeField(auto_now_add=True)
    duration_listened = models.IntegerField(default=0)

# ======================
# Mood Recommender AI
# ======================
class MoodRecommender:
    """მუსიკის რეკომენდერი განწყობის მიხედვით"""
    def __init__(self):
        self.model = None
        self.songs_df = None
        self.scaler = None
        self.mood_profiles = {
            'happy': {'energy': 0.8, 'valence': 0.9, 'danceability': 0.8},
            'sad': {'energy': 0.3, 'valence': 0.2, 'danceability': 0.3},
            'energetic': {'energy': 0.9, 'valence': 0.7, 'danceability': 0.9},
            'romantic': {'energy': 0.6, 'valence': 0.8, 'danceability': 0.6},
            'chill': {'energy': 0.4, 'valence': 0.6, 'danceability': 0.5},
            'angry': {'energy': 0.9, 'valence': 0.3, 'danceability': 0.4},
            'focused': {'energy': 0.5, 'valence': 0.5, 'danceability': 0.3},
            'party': {'energy': 0.9, 'valence': 0.9, 'danceability': 0.9},
            'workout': {'energy': 0.95, 'valence': 0.8, 'danceability': 0.8},
            'travel': {'energy': 0.7, 'valence': 0.7, 'danceability': 0.7},
        }

    def load_data(self):
        """Base მონაცემების DataFrame-ში ჩატვირთვა"""
        songs = Song.objects.all()
        data = []
        for song in songs:
            data.append({
                'id': song.id,
                'title': song.title,
                'artist': song.artist,
                'mood': song.mood,
                'genre': song.genre,
                'energy': float(song.energy),
                'valence': float(song.valence),
                'danceability': float(song.danceability),
                'tempo': float(song.tempo),
                'duration': float(song.duration),
                'youtube_link': song.youtube_link,
                'spotify_link': song.spotify_link,
            })
        self.songs_df = pd.DataFrame(data)
        return len(self.songs_df)

    def train_model(self):
        """KNN მოდელის ტრენინგი"""
        if self.songs_df is None or len(self.songs_df) < 1:
            return False
        features = self.songs_df[['energy','valence','danceability','tempo','duration']].values
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(features)
        n_neighbors = min(10, len(features))
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.model.fit(scaled_features)
        model_path = os.path.join(settings.AI_MODELS_DIR, 'mood_recommender.joblib')
        joblib.dump({'model': self.model, 'scaler': self.scaler}, model_path)
        return True

    def load_model(self):
        model_path = os.path.join(settings.AI_MODELS_DIR, 'mood_recommender.joblib')
        if os.path.exists(model_path):
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            return True
        return False

    def get_mood_profile(self, mood):
        return self.mood_profiles.get(mood, {'energy':0.5,'valence':0.5,'danceability':0.5})

    def recommend_by_mood(self, mood=None, exclude_ids=[], limit=10):
        """რეკომენდაცია კონკრეტული განწყობით"""
        if self.model is None or self.songs_df is None or len(self.songs_df)==0:
            return []

        mood_songs_df = self.songs_df
        if mood:
            mood_songs_df = self.songs_df[self.songs_df['mood']==mood]
        if len(mood_songs_df)==0:
            return []

        # Target vector
        mood_profile_vector = mood_songs_df[['energy','valence','danceability','tempo','duration']].mean().values.reshape(1,-1)
        scaled_target = self.scaler.transform(mood_profile_vector)

        n_neighbors = min(10, len(mood_songs_df))
        distances, indices = self.model.kneighbors(scaled_target, n_neighbors=n_neighbors)

        recommendations = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(mood_songs_df):
                continue
            song_data = mood_songs_df.iloc[idx]
            if song_data['id'] in exclude_ids:
                continue
            recommendations.append({
                'id': int(song_data['id']),
                'title': song_data['title'],
                'artist': song_data['artist'],
                'mood': song_data['mood'],
                'similarity': round(1 - distance,3),
                'youtube_link': song_data['youtube_link'],
                'spotify_link': song_data['spotify_link']
            })
            if len(recommendations)>=limit:
                break
        return recommendations

    def recommend_for_user(self, user_id, current_mood, limit=10):
        """რეკომენდაცია კონკრეტული მომხმარებლისთვის"""
        try:
            user = User.objects.get(id=user_id)
            listened_ids = list(PlayHistory.objects.filter(user=user).values_list('song_id', flat=True))
        except:
            listened_ids = []

        mood_recs = self.recommend_by_mood(current_mood, exclude_ids=listened_ids, limit=limit*2)
        if not mood_recs:
            mood_recs = self.recommend_by_mood(None, exclude_ids=listened_ids, limit=limit)
        return mood_recs

    def _get_user_preferences(self, user_id):
        """მომხმარებლის წინა შეფასებების ანალიზი"""
        try:
            ratings = UserRating.objects.filter(user_id=user_id)
            if ratings.count()>0:
                from collections import Counter
                mood_counts = ratings.values_list('mood_when_listened', flat=True)
                common_moods = [mood for mood,_ in Counter(mood_counts).most_common(3)]
                return {'preferred_moods': common_moods}
        except:
            pass
        return None
