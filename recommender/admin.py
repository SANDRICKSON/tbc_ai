# music_recommender/admin.py
from django.contrib import admin, messages
from django.http import HttpResponseRedirect, HttpResponse
from django import forms

import pandas as pd

from .models import Song, UserRating, PlayHistory, MoodRecommender


class CSVUploadForm(forms.Form):
    csv_file = forms.FileField(label="აირჩიეთ CSV ფაილი")


@admin.register(Song)
class SongAdmin(admin.ModelAdmin):
    list_display = ['title', 'artist', 'genre', 'mood', 'energy', 'valence']
    list_filter = ['genre', 'mood']
    search_fields = ['title', 'artist']
    fieldsets = [
        ('ძირითადი ინფორმაცია', {
            'fields': ['title', 'artist', 'genre', 'mood']
        }),
        ('მუსიკალური მახასიათებლები', {
            'fields': ['energy', 'valence', 'danceability', 'tempo', 'duration'],
            'description': '0-1 დიაპაზონში, სადაც 0=ძალიან დაბალი, 1=ძალიან მაღალი'
        }),
        ('ლინკები და ფაილები', {
            'fields': ['youtube_link', 'spotify_link', 'audio_file', 'cover_image']
        }),
    ]



    actions = ['export_to_csv', 'train_recommender']

    # ----------------------
    # CSV Export
    # ----------------------
    def export_to_csv(self, request, queryset):
        data = []
        for song in queryset:
            data.append({
                'Title': song.title,
                'Artist': song.artist,
                'Genre': song.get_genre_display(),
                'Mood': song.get_mood_display(),
                'Energy': song.energy,
                'Valence': song.valence,
                'Danceability': song.danceability,
                'Tempo': song.tempo,
                'Duration': song.duration,
                'YouTube': song.youtube_link,
                'Spotify': song.spotify_link,
            })
        df = pd.DataFrame(data)
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="songs_export.csv"'
        df.to_csv(response, index=False)
        return response

    export_to_csv.short_description = "ექსპორტი CSV-ში"

    # ----------------------
    # AI რეკომენდერის ტრენინგი
    # ----------------------
    def train_recommender(self, request, queryset):
        recommender = MoodRecommender()
        recommender.load_data()
        if recommender.train_model():
            self.message_user(request, "AI რეკომენდერი წარმატებით მოწონილია!")
        else:
            self.message_user(request, "ტრენინგისთვის საკმარისი მონაცემები არ არის", level=messages.ERROR)

    train_recommender.short_description = "AI რეკომენდერის ტრენინგი"

    # ----------------------
    # CSV Import + AI
    # ----------------------
    def get_urls(self):
        from django.urls import path
        urls = super().get_urls()
        custom_urls = [
            path('import-csv/', self.admin_site.admin_view(self.import_csv), name='songs_import_csv'),
        ]
        return custom_urls + urls

    def import_csv(self, request):
        if request.method == "POST":
            form = CSVUploadForm(request.POST, request.FILES)
            if form.is_valid():
                csv_file = form.cleaned_data['csv_file']
                try:
                    df = pd.read_csv(csv_file)
                    count_new = 0
                    for _, row in df.iterrows():
                        song, created = Song.objects.get_or_create(
                            title=row['Title'],
                            artist=row['Artist'],
                            defaults={
                                'genre': row['Genre'].lower(),
                                'mood': row['Mood'].lower(),
                                'energy': row['Energy'],
                                'valence': row['Valence'],
                                'danceability': row['Danceability'],
                                'tempo': row['Tempo'],
                                'duration': row['Duration'],
                                'youtube_link': row.get('YouTube', ''),
                                'spotify_link': row.get('Spotify', '')
                            }
                        )
                        if created:
                            count_new += 1

                    # AI რეკომენდერის ტრენინგი
                    recommender = MoodRecommender()
                    recommender.load_data()
                    recommender.train_model()

                    self.message_user(request, f"{count_new} ახალი სიმღერა დამატდა და AI რეკომენდერი მოწონილია!")
                    return HttpResponseRedirect("../")

                except Exception as e:
                    self.message_user(request, f"CSV-ის დამუშავების დროს მოხდა შეცდომა: {e}", level=messages.ERROR)
                    return HttpResponseRedirect("../")
        else:
            form = CSVUploadForm()

        context = dict(
            self.admin_site.each_context(request),
            form=form
        )
        from django.shortcuts import render
        return render(request, "admin/csv_form.html", context)


# ----------------------
# UserRating & PlayHistory Admin
# ----------------------
@admin.register(UserRating)
class UserRatingAdmin(admin.ModelAdmin):
    list_display = ['user', 'song', 'rating', 'mood_when_listened']
    list_filter = ['rating', 'mood_when_listened']


@admin.register(PlayHistory)
class PlayHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'song', 'listened_at']
    list_filter = ['listened_at']
