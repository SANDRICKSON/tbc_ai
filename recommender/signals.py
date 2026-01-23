# recommender/signals.py
from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from .models import Song, UserRating
import threading
import time
import os
from django.conf import settings
import joblib
from .ml_model import MoodRecommender, get_mood_recommender

def retrain_model_delayed():
    """დაყოვნებული მოდელის განახლება"""
    def retrain():
        # დაელოდეთ რამდენიმე წამს, რომ დამატებული/წაშლილი სიმღერები დარეგისტრირდეს ბაზაში
        time.sleep(3)
        
        try:
            from .models import Song
            recommender = get_mood_recommender()
            
            # ყველა სიმღერის მიღება
            songs = list(Song.objects.all())
            print(f"🔄 მოდელის განახლება... ბაზაში სიმღერები: {len(songs)}")
            
            if songs:
                if recommender.train(songs):
                    # მოდელის შენახვა ფაილში
                    model_path = os.path.join(settings.BASE_DIR, 'mood_recommender.pkl')
                    recommender.save_model(model_path)
                    print(f"✅ მოდელი განახლდა: {len(songs)} სიმღერა")
                else:
                    print("⚠️ მოდელის განახლება ვერ მოხერხდა")
            else:
                print("ℹ️ ბაზაში სიმღერები არ არის")
                
        except Exception as e:
            print(f"❌ მოდელის განახლების შეცდომა: {e}")
    
    # გაუშვით ცალკე თრედში
    thread = threading.Thread(target=retrain)
    thread.daemon = True
    thread.start()

@receiver(post_save, sender=Song)
def song_saved(sender, instance, created, **kwargs):
    """სიმღერის დამატების ან რედაქტირებისას"""
    print(f"🎵 სიმღერა {'დაემატა' if created else 'განახლდა'}: {instance.title}")
    retrain_model_delayed()

@receiver(post_delete, sender=Song)
def song_deleted(sender, instance, **kwargs):
    """სიმღერის წაშლისას"""
    print(f"🗑️ სიმღერა წაიშალა: {instance.title}")
    retrain_model_delayed()

@receiver(post_save, sender=UserRating)
def rating_saved(sender, instance, created, **kwargs):
    """რეიტინგის დამატებისას"""
    print(f"⭐ რეიტინგი დაემატა: {instance.user.username} -> {instance.song.title}")
    # აქაც შეგიძლიათ მოდელის განახლება, თუ რეიტინგებითაც ტრენინგდება
