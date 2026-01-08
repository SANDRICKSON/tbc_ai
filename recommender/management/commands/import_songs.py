# music_recommender/management/commands/import_songs.py
import pandas as pd
from django.core.management.base import BaseCommand

from recommender.models import MoodRecommender, Song


class Command(BaseCommand):
    help = 'სიმღერების იმპორტი CSV ფაილიდან'

    def add_arguments(self, parser):
        parser.add_argument('csv_file', type=str, help='CSV ფაილის მისამართი')

    def handle(self, *args, **kwargs):
        csv_file = kwargs['csv_file']

        try:
            df = pd.read_csv(csv_file)
            imported = 0
            skipped = 0

            for _, row in df.iterrows():
                # ვამოწმებთ, არსებობს თუ არა უკვე
                if Song.objects.filter(title=row['title'], artist=row['artist']).exists():
                    skipped += 1
                    continue

                # ახალი სიმღერის შექმნა
                song = Song(
                    title=row['title'],
                    artist=row['artist'],
                    genre=row.get('genre', 'pop'),
                    mood=row.get('mood', 'happy'),
                    energy=float(row.get('energy', 0.5)),
                    valence=float(row.get('valence', 0.5)),
                    danceability=float(row.get('danceability', 0.5)),
                    tempo=int(row.get('tempo', 120)),
                    duration=int(row.get('duration', 180)),
                    youtube_link=row.get('youtube_link', ''),
                    spotify_link=row.get('spotify_link', ''),
                )
                song.save()
                imported += 1

            self.stdout.write(
                self.style.SUCCESS(f'წარმატებით იმპორტირდა {imported} სიმღერა, გამოტოვდა {skipped}')
            )

            # AI მოდელის ტრენინგი

            recommender = MoodRecommender()
            recommender.load_data()
            if recommender.train_model():
                self.stdout.write(self.style.SUCCESS('AI რეკომენდერი წარმატებით იქნა მოწონებული!'))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f'შეცდომა: {str(e)}'))