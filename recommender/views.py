# music_recommender/views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
import json

from .models import Song, UserRating, PlayHistory, MoodRecommender
from .forms import RegisterForm, LoginForm


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

            # MoodRecommender ინიციალიზაცია და ტრენინგი
            recommender = MoodRecommender()
            recommender.load_data()
            recommender.train_model()  # ყოველთვის ტრენინგი ახალი მონაცემების მიხედვით

            # რეკომენდაციების მიღება
            if user_id:
                recommendations = recommender.recommend_for_user(user_id, current_mood, limit)
            else:
                recommendations = recommender.recommend_by_mood(current_mood, limit=limit)

            # სიმღერების დეტალური ინფორმაცია
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
                    continue

            # fallback: თუ არაფერია რეკომენდაცია, პირდაპირ აიღე სიმღერები mood-ით
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
                'recommendations': detailed_recommendations
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'არასწორი მეთოდი'}, status=405)


@login_required
def rate_song(request):
    """სიმღერის შეფასება"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            song_id = data.get('song_id')
            rating = data.get('rating')
            mood = data.get('mood')

            song = get_object_or_404(Song, id=song_id)

            # შეფასების შენახვა ან განახლება
            user_rating, created = UserRating.objects.update_or_create(
                user=request.user,
                song=song,
                defaults={'rating': rating, 'mood_when_listened': mood}
            )

            return JsonResponse({
                'success': True,
                'message': 'შეფასება შენახულია',
                'rating_id': user_rating.id
            })

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'არასწორი მეთოდი'}, status=405)


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
        } for song in songs]
    })


# ========================
# ავტორიზაცია და რეგისტრაცია
# ========================

def register_view(request):
    """მომხმარებლის რეგისტრაცია"""
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            login(request, user)  # ავტომატურად შეხვიდე რეგისტრაციის შემდეგ
            return redirect('index')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('index')  # მთავარი გვერდი
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    """გასვლა"""
    logout(request)
    return redirect('index')
