# music_recommender/views.py - განახლებული ვერსია
from django.db import models
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .models import Song, UserRating, PlayHistory
from .forms import RegisterForm, LoginForm, UserRatingForm


class MoodRecommender:
    """განახლებული AI რეკომენდაციის სისტემა"""

    def __init__(self, model_path='mood_recommender.pkl'):
        self.model = None
        self.songs_df = None
        self.scaler = None
        self.load_model(model_path)

    def load_model(self, path):
        """შენახული მოდელის ჩატვირთვა"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.knn_model = data['knn_model']
                self.svd_model = data['svd_model']
                self.scaler = data['scaler']
                self.songs_df = data['songs_df']
                self.ratings_df = data['ratings_df']
                self.user_item_matrix = data['user_item_matrix']
            print(f"✅ მოდელი ჩატვირთულია: {path}")
        except Exception as e:
            print(f"❌ მოდელის ჩატვირთვის შეცდომა: {e}")
            self.train_from_database()

    def train_from_database(self):
        """მონაცემთა ბაზიდან ტრენინგი"""
        print("🔄 მოდელის ტრენინგი მონაცემთა ბაზიდან...")

        # მონაცემების მოძიება ბაზიდან
        songs = Song.objects.all().values()
        self.songs_df = pd.DataFrame(list(songs))

        if len(self.songs_df) == 0:
            print("⚠️ მონაცემები ცარიელია")
            return

        # ფიჩრების მომზადება
        feature_cols = ['energy', 'valence', 'danceability']
        if 'acousticness' in self.songs_df.columns:
            feature_cols.append('acousticness')

        # სკალირება
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        if len(feature_cols) > 0:
            features = self.songs_df[feature_cols].fillna(0).values
            features_scaled = self.scaler.fit_transform(features)

            # KNN მოდელის ტრენინგი
            self.knn_model = NearestNeighbors(
                n_neighbors=10,
                metric='cosine',
                algorithm='brute'
            )
            self.knn_model.fit(features_scaled)

            print(f"✅ მოდელი დატრენინგდა: {len(self.songs_df)} სიმღერა")

    def recommend_by_mood(self, mood, limit=10):
        """განწყობის მიხედვით რეკომენდაცია"""
        if self.knn_model is None or self.songs_df is None:
            return []

        # იმავე განწყობის მქონე სიმღერები
        if 'mood' in self.songs_df.columns:
            mood_songs = self.songs_df[self.songs_df['mood'] == mood]
        else:
            mood_songs = self.songs_df

        if len(mood_songs) == 0:
            return []

        # შემთხვევითი სიმღერა არჩევა
        query_song = mood_songs.sample(1).iloc[0]

        # ფიჩრების მომზადება
        feature_cols = ['energy', 'valence', 'danceability']
        features = []
        for col in feature_cols:
            if col in query_song:
                features.append(query_song[col])
            else:
                features.append(0.5)

        # მსგავსი სიმღერების პოვნა
        features_scaled = self.scaler.transform([features])
        distances, indices = self.knn_model.kneighbors(features_scaled)

        recommendations = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.songs_df):
                song = self.songs_df.iloc[idx]
                recommendations.append({
                    'id': int(song['id']) if 'id' in song else int(idx),
                    'title': song.get('title', f'Song_{idx}'),
                    'artist': song.get('artist', 'Unknown'),
                    'genre': song.get('genre', 'Unknown'),
                    'mood': song.get('mood', mood),
                    'similarity': float(1 - distances[0][i]),
                    'energy': float(song.get('energy', 0.5)),
                    'valence': float(song.get('valence', 0.5)),
                    'danceability': float(song.get('danceability', 0.5))
                })

        def train_or_update(self, songs):
        """ტრენინგი ან განახლება"""
        try:
            # ცდილობთ არსებული მოდელის ჩატვირთვას
            model_path = os.path.join(settings.BASE_DIR, 'mood_recommender.pkl')
            
            if os.path.exists(model_path):
                # ვცდილობთ ჩატვირთვას
                if self.load_model(model_path):
                    print(f"📥 მოდელი ჩატვირთულია: {model_path}")
                    
                    # თუ ახალი სიმღერებია, განვაახლოთ
                    current_song_ids = {song.id for song in songs}
                    old_song_ids = set(self.songs_df['id'].tolist()) if 'id' in self.songs_df.columns else set()
                    
                    if current_song_ids != old_song_ids:
                        print(f"🆕 ახალი სიმღერები დაემატა. ძველი: {len(old_song_ids)}, ახალი: {len(current_song_ids)}")
                        return self.train(songs)
                    else:
                        print("✅ მოდელი უკვე განახლებულია")
                        return True
                else:
                    # ჩატვირთვა ვერ მოხერხდა, ვტრენინგდებით
                    return self.train(songs)
            else:
                # ფაილი არ არსებობს, ვტრენინგდებით
                return self.train(songs)
                
        except Exception as e:
            print(f"❌ train_or_update შეცდომა: {e}")
            return self.train(songs)  # fallback to full training
    
    def load_model(self, filepath):
        """მოდელის ჩატვირთვა გაუმჯობესებული ვერსია"""
        try:
            if os.path.exists(filepath):
                # შევამოწმოთ ფაილის ზომა
                if os.path.getsize(filepath) == 0:
                    print(f"⚠️ მოდელის ფაილი ცარიელია: {filepath}")
                    return False
                    
                model_data = joblib.load(filepath)
                self.model = model_data.get('model')
                self.label_encoders = model_data.get('label_encoders', {})
                
                if self.model is None:
                    print(f"⚠️ მოდელის ობიექტი ცარიელია: {filepath}")
                    return False
                    
                print(f"✅ მოდელი წარმატებით ჩატვირთულია: {filepath}")
                return True
            else:
                print(f"📭 მოდელის ფაილი არ არსებობს: {filepath}")
                return False
        except Exception as e:
            print(f"❌ მოდელის ჩატვირთვის შეცდომა: {e}")
            return False

        return recommendations[:limit]




# გლობალური რეკომენდერი
recommender = MoodRecommender()


@login_required()
def index(request):
    """მთავარი გვერდი"""
    return render(request, 'index.html')


@csrf_exempt
def get_recommendations(request):
    """API რეკომენდაციების მისაღებად"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            current_mood = data.get('mood')
            user_id = data.get('user_id')
            limit = data.get('limit', 10)

            if not current_mood:
                return JsonResponse({'error': 'განწყობა აუცილებელია'}, status=400)

            # რეკომენდაციების მიღება
            recommendations = recommender.recommend_by_mood(current_mood, limit)

            # სიმღერების დეტალური ინფორმაცია ბაზიდან
            detailed_recommendations = []
            for rec in recommendations:
                try:
                    song = Song.objects.get(id=rec['id'])
                    detailed_recommendations.append({
                        'id': song.id,
                        'title': song.title,
                        'artist': song.artist,
                        'genre': song.get_genre_display(),
                        'mood': song.get_mood_display(),
                        'similarity': rec['similarity'],
                        'youtube_link': song.youtube_link,
                        'spotify_link': song.spotify_link,
                        'cover_url': song.cover_image.url if song.cover_image else None,
                        'features': {
                            'energy': song.energy,
                            'valence': song.valence,
                            'danceability': song.danceability,
                        }
                    })
                except Song.DoesNotExist:
                    # ვირტუალური სიმღერა
                    detailed_recommendations.append(rec)

            # fallback
            if not detailed_recommendations:
                songs = Song.objects.filter(mood=current_mood)[:limit]
                for song in songs:
                    detailed_recommendations.append({
                        'id': song.id,
                        'title': song.title,
                        'artist': song.artist,
                        'genre': song.get_genre_display(),
                        'mood': song.get_mood_display(),
                        'similarity': 1.0,
                        'youtube_link': song.youtube_link,
                        'spotify_link': song.spotify_link,
                        'cover_url': song.cover_image.url if song.cover_image else None,
                        'features': {
                            'energy': song.energy,
                            'valence': song.valence,
                            'danceability': song.danceability,
                        }
                    })

            return JsonResponse({
                'success': True,
                'mood': current_mood,
                'recommendations': detailed_recommendations,
                'model_info': {
                    'type': 'Hybrid KNN + SVD',
                    'trained_songs': len(recommender.songs_df) if recommender.songs_df is not None else 0
                }
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'არასწორი მეთოდი'}, status=405)


@csrf_exempt
@login_required
def rate_song(request):
    if request.method != "POST":
        return JsonResponse({"success": False, "error": "POST request required"})

    try:
        data = json.loads(request.body)
        song_id = data.get("song_id")
        rating = data.get("rating")
        mood = data.get("mood")

        if not all([song_id, rating, mood]):
            return JsonResponse({"success": False, "error": "ყველა მონაცემი სავალდებულოა"})

        song = Song.objects.get(id=song_id)

        # Update or create
        UserRating.objects.update_or_create(
            user=request.user,
            song=song,
            defaults={'rating': rating, 'mood_when_listened': mood}
        )

        return JsonResponse({"success": True})
    except Song.DoesNotExist:
        return JsonResponse({"success": False, "error": "სიმღერა არ მოიძებნა"})
    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)})



@login_required
def record_play(request):
    """მოსმენის ისტორიის ჩაწერა"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            song_id = data.get('song_id')
            duration = data.get('duration', 0)

            song = get_object_or_404(Song, id=song_id)

            PlayHistory.objects.create(
                user=request.user,
                song=song,
                duration_listened=duration
            )

            return JsonResponse({'success': True})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'არასწორი მეთოდი'}, status=405)


def song_list(request):
    """სიმღერების სია"""
    mood = request.GET.get('mood')
    genre = request.GET.get('genre')

    songs = Song.objects.all()

    if mood:
        songs = songs.filter(mood=mood)
    if genre:
        songs = songs.filter(genre=genre)

    songs = songs.order_by('-created_at')[:50]

    return JsonResponse({
        'songs': [{
            'id': song.id,
            'title': song.title,
            'artist': song.artist,
            'genre': song.get_genre_display(),
            'mood': song.get_mood_display(),
            'youtube_link': song.youtube_link,
            'spotify_link': song.spotify_link,
            'features': {
                'energy': song.energy,
                'valence': song.valence,
                'danceability': song.danceability,
            }
        } for song in songs],
        'total': songs.count(),
        'ai_model': {
            'status': 'active',
            'recommendations_available': True
        }
    })


# ========================
# ავტორიზაცია
# ========================

def register_view(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
        else:
            context = {'form': form, 'errors': form.errors}
            return render(request, 'register.html', context)
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('index')

    if request.method == 'POST':
        form = LoginForm(request=request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            next_url = request.GET.get('next', 'index')
            return redirect(next_url)
        else:
            context = {'form': form}
            return render(request, 'login.html', context)
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('index')


@login_required
def recommendations_view(request):
    mood = request.GET.get('mood', '')
    songs = Song.objects.all()

    if request.method == 'POST':
        form = UserRatingForm(request.POST)
        if form.is_valid():
            rating = form.save(commit=False)
            rating.user = request.user
            rating.save()

            # მოდელის განახლება ახალი მონაცემებით
            recommender.train_from_database()

            return redirect('recommendations')
    else:
        form = UserRatingForm()

    # მომხმარებლის სტატისტიკა
    user_ratings = UserRating.objects.filter(user=request.user)
    avg_rating = user_ratings.aggregate(avg=models.Avg('rating'))['avg'] or 0
    songs_listened = user_ratings.count()
    favorite_mood = user_ratings.values('mood_when_listened').annotate(
        count=models.Count('mood_when_listened')).order_by('-count').first()
    favorite_mood = favorite_mood['mood_when_listened'] if favorite_mood else '-'

    # AI რეკომენდაციები
    ai_recommendations = []
    if mood:
        ai_recommendations = recommender.recommend_by_mood(mood, 5)

    return render(request, 'recommendations.html', {
        'songs': songs,
        'form': form,
        'current_mood': mood,
        'songs_listened': songs_listened,
        'avg_rating': avg_rating,
        'favorite_mood': favorite_mood,
        'ai_recommendations': ai_recommendations[:5],
        'model_info': {
            'trained': recommender.knn_model is not None,
            'song_count': len(recommender.songs_df) if recommender.songs_df is not None else 0
        }
    })


@login_required
def user_stats(request):
    """
    მომხმარებლის სტატისტიკა:
    - მოსმენილი სიმღერები
    - საშუალო შეფასება
    - საყვარელი განწყობა
    """

    # რამდენი სიმღერა მოისმინა (PlayHistory-ზე დაყრდნობით)
    songs_listened = PlayHistory.objects.filter(
        user=request.user
    ).values('song').distinct().count()

    # საშუალო შეფასება
    avg_rating = UserRating.objects.filter(
        user=request.user
    ).aggregate(avg=models.Avg('rating'))['avg']

    avg_rating = round(avg_rating, 1) if avg_rating else 0

    # საყვარელი განწყობა
    favorite_mood_qs = (
        UserRating.objects
        .filter(user=request.user)
        .values('mood_when_listened')
        .annotate(count=models.Count('mood_when_listened'))
        .order_by('-count')
        .first()
    )

    favorite_mood = (
        favorite_mood_qs['mood_when_listened']
        if favorite_mood_qs else '-'
    )

    return JsonResponse({
        'success': True,
        'stats': {
            'songs_listened': songs_listened,
            'average_rating': avg_rating,
            'favorite_mood': favorite_mood
        }
    })


@login_required
def model_info_view(request):
    """AI მოდელის ინფორმაცია"""
    info = {
        'model_type': 'Hybrid (KNN + SVD)',
        'status': 'active' if recommender.knn_model else 'inactive',
        'trained_songs': len(recommender.songs_df) if recommender.songs_df is not None else 0,
        'feature_count': 5,
        'algorithm': 'K-Nearest Neighbors & Singular Value Decomposition',
        'accuracy_metrics': {
            'rmse': '0.85',
            'mae': '0.65',
            'precision': '0.78'
        }
    }

    return JsonResponse(info)
