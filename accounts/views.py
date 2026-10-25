from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages

from .forms import SignUpForm


# ==========================================
# Home Page
# ==========================================

def home(request):

    return render(
        request,
        "home.html"
    )


# ==========================================
# Sign Up
# ==========================================

def signup_view(request):

    if request.method == "POST":

        form = SignUpForm(
            request.POST
        )

        if form.is_valid():

            form.save()

            messages.success(
                request,
                "Account created successfully! Please login."
            )

            return redirect(
                "login"
            )

    else:

        form = SignUpForm()

    return render(
        request,
        "accounts/signup.html",
        {
            "form": form
        }
    )


# ==========================================
# Login
# ==========================================

def login_view(request):

    if request.method == "POST":

        username = request.POST.get(
            "username"
        )

        password = request.POST.get(
            "password"
        )

        user = authenticate(
            request,
            username=username,
            password=password
        )

        if user is not None:

            login(
                request,
                user
            )

            return redirect(
                "profile"
            )

        else:

            messages.error(
                request,
                "Invalid username or password."
            )

    return render(
        request,
        "accounts/login.html"
    )


# ==========================================
# Logout
# ==========================================

def logout_view(request):

    logout(request)

    messages.success(
        request,
        "Logged out successfully!"
    )

    return redirect(
        "home"
    )
    