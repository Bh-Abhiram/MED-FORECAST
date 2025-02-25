from django.contrib import admin
from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('heart/', views.heart, name="heart"),
    path('diabetes/', views.diabetes, name="diabetes"),
    path('breast/', views.breast, name="breast"),
    path('bmi/', views.bmi, name="bmi"),
    path('blood_pressure/', views.blood_pressure, name="blood_pressure"),
    path('covid_19/', views.covid_19, name="covid_19"),
    path('', views.home, name="home"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
