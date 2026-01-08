from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm
from django.core.exceptions import ValidationError

class RegisterForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput, label="პაროლი")
    password_confirm = forms.CharField(widget=forms.PasswordInput, label="პაროლის დადასტურება")

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'last_name']

    def clean_password_confirm(self):
        password = self.cleaned_data.get('password')
        password_confirm = self.cleaned_data.get('password_confirm')
        if password != password_confirm:
            raise ValidationError("პაროლები არ ემთხვევა!")
        return password_confirm


class LoginForm(AuthenticationForm):
    username = forms.CharField(label="სახელი")
    password = forms.CharField(widget=forms.PasswordInput, label="პაროლი")
