from django.shortcuts import render

# Create your views here.
def form(requests):
    return render(requests, 'form/welcome.html')