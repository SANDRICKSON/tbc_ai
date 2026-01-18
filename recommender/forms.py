from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.core.exceptions import ValidationError

from recommender.models import UserRating


class RegisterForm(UserCreationForm):
    email = forms.EmailField(label="ელ. ფოსტა", required=True)
    first_name = forms.CharField(label="სახელი", required=False)
    last_name = forms.CharField(label="გვარი", required=False)

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name', 'password1', 'password2']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['username'].label = "მომხმარებელი"
        self.fields['password1'].label = "პაროლი"
        self.fields['password2'].label = "პაროლის დადასტურება"
        # Add Bootstrap classes
        for field in self.fields:
            self.fields[field].widget.attrs.update({'class': 'form-control'})

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise ValidationError("ეს ელ. ფოსტა უკვე რეგისტრირებულია!")
        return email

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if User.objects.filter(username=username).exists():
            raise ValidationError("ეს მომხმარებელი უკვე არსებობს!")
        return username


class LoginForm(AuthenticationForm):
    username = forms.CharField(label="მომხმარებელი", widget=forms.TextInput(attrs={'class': 'form-control'}))
    password = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}), label="პაროლი")


class UserRatingForm(forms.ModelForm):
    class Meta:
        model = UserRating
        fields = ['song', 'rating', 'mood_when_listened']
        widgets = {
            'song': forms.HiddenInput(),  # სიმღერა ფიქსირებული იქნება JS-ით
            'rating': forms.Select(choices=[(i, i) for i in range(1, 6)], attrs={'class': 'form-select'}),
            'mood_when_listened': forms.Select(attrs={'class': 'form-select'}),
        }