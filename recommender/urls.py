# music_recommender/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # მთავარი გვერდი
    path('', views.index, name='index'),

    # რეგისტრაცია, ლოგინი, ლოგაუთი
    path('api/register/', views.register_view, name='register'),
    path('api/login/', views.login_view, name='login'),
    path('api/logout/', views.logout_view, name='logout'),

    # API endpoints
    path('api/recommend/', views.get_recommendations, name='get_recommendations'),
    path('api/rate/', views.rate_song, name='rate_song'),
    path('api/play/', views.record_play, name='record_play'),
    path('api/songs/', views.song_list, name='song_list'),
    path('api/user-stats/', views.user_stats, name='user_stats'),

]
